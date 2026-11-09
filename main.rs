// =============================================================================
// 選擇權策略蒙地卡羅模擬（Rust 單檔版，移植自 R v3；無外部依賴）
//
// 編譯：  rustc -O main.rs -o sim
// 逐日回放：./sim --replay [--replay-out replay.csv]（依序只用前 1..N 個實際收盤各模擬一次，看情境機率如何隨每天收盤變化）
// 執行：  ./sim [--config config.txt] [--out heatmap.html] [--no-heavy] [--sparse] [--cores N] [--print-config]
//
// 結構（與 R 版相同）：
//   PART 1  引擎（封閉）：定價迴圈、出場評估、run()、setup_trade()
//   PART 2  外掛模組（開放）：定價器、路徑模型、出場規則、情境分類、報告（REPORTS 註冊表）
//   PART 3  交易定義（每個新部位只改 trade()）
//   PART 4  main：執行與輸出
//
// 擴充：新部位 → 改 trade()；新報告 → 在 reports() 加一筆 ReportDef；
//       新出場規則 → 在 build_exit() 加一個 Kind；新分類 → 在 OutcomeKind 加一個分支。
// 注意：亂數器為 xoshiro256++，與 R 的 Mersenne Twister 不同，個別路徑不會逐條相同，
//       但統計結果在抽樣誤差內應一致。
// =============================================================================

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

type Day = i32; // 自 1970-01-01 起算的天數

static NCORES: AtomicUsize = AtomicUsize::new(1);
static QUIET: AtomicBool = AtomicBool::new(false); // 回放模式時關閉 setup_trade 的逐行輸出
fn quiet() -> bool { QUIET.load(Ordering::Relaxed) }

// ---------------------------------------------------------------------------
// 日期工具
// ---------------------------------------------------------------------------
fn days_from_civil(y: i32, m: u32, d: u32) -> Day {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = (y - era * 400) as u32;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe as i32 - 719468
}
fn civil_from_days(z: Day) -> (i32, u32, u32) {
    let z = z + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i32 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}
fn days_in_month(y: i32, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 { 29 } else { 28 },
        _ => 0,
    }
}
// 嚴格解析 YYYY-MM-DD（只看位元組、不配置記憶體，方便形式化驗證）；日期不存在（例如 2026-02-30）回傳 None
fn parse_ymd(b: &[u8]) -> Option<(i32, u32, u32)> {
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' { return None; }
    let two = |i: usize| -> Option<u32> {
        if b[i].is_ascii_digit() && b[i + 1].is_ascii_digit() { Some((b[i] - b'0') as u32 * 10 + (b[i + 1] - b'0') as u32) } else { None }
    };
    let (hi, lo, m, d) = (two(0)?, two(2)?, two(5)?, two(8)?);
    let y = (hi * 100 + lo) as i32;
    if m < 1 || m > 12 || d < 1 || d > days_in_month(y, m) { return None; }
    Some((y, m, d))
}
fn try_date(s: &str) -> Result<Day, String> {
    match parse_ymd(s.as_bytes()) {
        Some((y, m, d)) => Ok(days_from_civil(y, m, d)),
        None => Err(format!("日期格式應為 YYYY-MM-DD 且必須是存在的日期: {}", s)),
    }
}
fn date(s: &str) -> Day {
    try_date(s.trim()).unwrap_or_else(|e| panic!("{}", e))
}
fn fmt_md(d: Day) -> String {
    let (_, m, dd) = civil_from_days(d);
    format!("{:02}/{:02}", m, dd)
}
fn iso_weekday(d: Day) -> i32 {
    (d + 3).rem_euclid(7) + 1 // 1 = 週一 … 7 = 週日
}

// ---------------------------------------------------------------------------
// 亂數與統計
// ---------------------------------------------------------------------------
struct Rng {
    s: [u64; 4],
    spare: Option<f64>,
}
impl Rng {
    fn new(seed: u64) -> Self {
        let mut x = seed;
        let mut s = [0u64; 4];
        for v in s.iter_mut() {
            x = x.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            *v = z ^ (z >> 31);
        }
        Rng { s, spare: None }
    }
    fn next_u64(&mut self) -> u64 {
        let r = self.s[0].wrapping_add(self.s[3]).rotate_left(23).wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }
    fn unif(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn norm(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        let u1 = self.unif();
        let u2 = self.unif();
        let r = (-2.0 * u1.ln()).sqrt();
        let th = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some(r * th.sin());
        r * th.cos()
    }
}

fn mean(x: &[f64]) -> f64 {
    if x.is_empty() { f64::NAN } else { x.iter().sum::<f64>() / x.len() as f64 }
}
fn sd(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 { return f64::NAN; }
    let m = mean(x);
    (x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (n as f64 - 1.0)).sqrt()
}
fn quantile(x: &[f64], p: f64) -> f64 {
    // R type 7
    let mut v = x.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let h = (v.len() as f64 - 1.0) * p;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    v[lo] + (h - lo as f64) * (v[hi] - v[lo])
}
fn median(x: &[f64]) -> f64 { quantile(x, 0.5) }

fn par_map<T: Send, F: Fn(usize) -> T + Sync>(n: usize, f: F) -> Vec<T> {
    let nc = NCORES.load(Ordering::Relaxed).max(1);
    if nc <= 1 || n < 2 {
        return (0..n).map(&f).collect();
    }
    let next = AtomicUsize::new(0);
    let mut parts: Vec<Vec<(usize, T)>> = Vec::new();
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..nc.min(n))
            .map(|_| {
                sc.spawn(|| {
                    let mut v = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n { break; }
                        v.push((i, f(i)));
                    }
                    v
                })
            })
            .collect();
        for h in hs { parts.push(h.join().unwrap()); }
    });
    let mut out: Vec<Option<T>> = (0..n).map(|_| None).collect();
    for p in parts { for (i, x) in p { out[i] = Some(x); } }
    out.into_iter().map(|x| x.unwrap()).collect()
}

// ---------------------------------------------------------------------------
// 表格輸出
// ---------------------------------------------------------------------------
struct Table {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}
fn disp_w(s: &str) -> usize {
    s.chars().map(|c| if (c as u32) >= 0x2E80 { 2 } else { 1 }).sum()
}
impl Table {
    fn print(&self) {
        let nc = self.headers.len();
        let mut w: Vec<usize> = self.headers.iter().map(|h| disp_w(h)).collect();
        for r in &self.rows {
            for (j, c) in r.iter().enumerate().take(nc) { w[j] = w[j].max(disp_w(c)); }
        }
        let line = |cells: &Vec<String>| {
            let mut s = String::new();
            for (j, c) in cells.iter().enumerate().take(nc) {
                let pad = w[j] - disp_w(c);
                s.push_str(&" ".repeat(pad));
                s.push_str(c);
                s.push_str("  ");
            }
            println!("{}", s.trim_end());
        };
        line(&self.headers);
        for r in &self.rows { line(r); }
    }
}
fn f1(x: f64) -> String { if x.is_nan() { "NA".into() } else { format!("{:.1}", x) } }
fn f2(x: f64) -> String { if x.is_nan() { "NA".into() } else { format!("{:.2}", x) } }
fn f3(x: f64) -> String { if x.is_nan() { "NA".into() } else { format!("{:.3}", x) } }

// =============================================================================
// PART 1：核心引擎
// =============================================================================
#[derive(Clone, Copy)]
struct Env {
    r: f64,
    sigma: f64,
    skew: f64,
    s0_ref: f64,
    vf: f64,
    term_a: f64,
    term_tau: f64,
}

#[derive(Clone)]
struct Leg { qty: f64, k: f64, is_call: bool, expiry: Day, iv: f64 }
#[derive(Clone)]
struct Strategy { debit: f64, legs: Vec<Leg> }

#[derive(Clone, Copy, PartialEq)]
enum PricerKind { Bs, BsDyn }
#[derive(Clone, Copy, PartialEq)]
enum PathKind { Normal, Sv }

fn pricer_by_name(n: &str) -> PricerKind {
    match n { "bs" => PricerKind::Bs, "bs_dyn" => PricerKind::BsDyn, _ => panic!("未註冊定價器: {}", n) }
}

#[derive(Clone)]
struct Path { dates: Vec<Day>, s: Vec<f64>, vf: Option<Vec<f64>> }
#[derive(Clone)]
struct Res { dates: Vec<Day>, s: Vec<f64>, pnl: Vec<f64> }
impl Res {
    fn last_pnl(&self) -> f64 { *self.pnl.last().unwrap() }
}

struct State { date: Day, s: f64, pnl: f64, hwm: f64 } // hwm = 建倉以來（含實際收盤）到目前為止的最高收盤
type ExitFn = Box<dyn Fn(&State) -> bool + Send + Sync>;

#[derive(Clone, Copy)]
struct SeedFn { base: u64, mult: u64 }

#[derive(Clone, Copy)]
struct Sv { kappa: f64, xi: f64, rho: f64 }

#[derive(Clone)]
struct Cfg {
    stop_price: f64, swan_price: f64, swan_extra_loss: f64, bull_price: f64,
    swan_loss_from_today: bool, time_stop_date: Day,
    jump_lambda: f64, jump_size: f64, open_shock_sd: f64,
    pnl_big: f64,
    atr_k: Option<f64>, atr_value: f64,
    horizon: Day, pnl_today: f64, swan_pnl0: f64, // 由 setup_trade 填入
}

#[derive(Clone)]
struct Trendline { name: String, d1: Day, p1: f64, d2: Day, p2: f64 }
fn trend_at(tl: &Trendline, d: Day) -> f64 {
    let slope = (tl.p2 - tl.p1) / (tl.d2 - tl.d1) as f64;
    tl.p1 + slope * (d - tl.d1) as f64
}

// ---- 出場規則規格 -----------------------------------------------------------------
#[derive(Clone)]
enum StopArg { Default, Price(f64), None }
#[derive(Clone)]
enum Kind { Ladder, TakeProfit(f64), AnyOf(Vec<RuleSpec>) }
#[derive(Clone)]
struct RuleSpec { name: String, kind: Kind, stop: StopArg, time_stop: bool, breaker: bool, window_tp: Option<f64>, atr: Option<f64> }
impl RuleSpec {
    fn new(name: &str) -> Self {
        RuleSpec { name: name.into(), kind: Kind::Ladder, stop: StopArg::Default, time_stop: true, breaker: true, window_tp: None, atr: None }
    }
    fn tp(name: &str, target: f64) -> Self {
        let mut s = Self::new(name);
        s.kind = Kind::TakeProfit(target);
        s
    }
    fn stop(mut self, p: f64) -> Self { self.stop = StopArg::Price(p); self }
    fn no_stop(mut self) -> Self { self.stop = StopArg::None; self }
    fn no_time_stop(mut self) -> Self { self.time_stop = false; self }
    fn no_breaker(mut self) -> Self { self.breaker = false; self }
    // 時間停損日起，總損益落在 0 ~ cap 之間就止盈出場（損益 < 0 仍由時間停損處理）
    fn window_tp(mut self, cap: f64) -> Self { self.window_tp = Some(cap); self }
    // 吊燈式 ATR 停損：收盤 < 建倉以來最高收盤 - k × ATR（ATR 值取 cfg.atr_value）
    fn atr(mut self, k: f64) -> Self { self.atr = Some(k); self }
}

#[derive(Clone)]
struct Scenario {
    name: String, pricer: PricerKind, path: PathKind,
    skew: Option<f64>, vol: Option<f64>, iv_scale: f64, term_a: f64, term_tau: f64,
}
impl Scenario {
    fn new(name: &str) -> Self {
        Scenario { name: name.into(), pricer: PricerKind::Bs, path: PathKind::Normal,
                   skew: None, vol: None, iv_scale: 1.0, term_a: 0.0, term_tau: 0.1 }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Objective { Mean, LossAverse }
#[derive(Clone, Copy, PartialEq)]
enum Prefer { Higher, Lower, Best }
#[derive(Clone)]
struct AutoTp {
    grid: Vec<f64>, objective: Objective, lambda: f64, slippage: f64, se_k: f64, prefer: Prefer,
}

#[derive(Clone, Copy)]
struct HoldParams { swan_pnl0: f64, swan_extra_loss: f64, horizon: Day, pnl_big: f64, atr_k: Option<f64>, atr_value: f64 }
// 持有型情境分類（0 大幅獲利 1 小幅獲利 2 到期小幅虧損 3 時間停損 4 虧損熔斷 5 區間止盈 6 ATR停損）；純函數，可形式化驗證
fn classify_hold(p: &HoldParams, s_end: f64, pnl_end: f64, d_end: Day, hwm: f64) -> usize {
    if pnl_end < p.swan_pnl0 - p.swan_extra_loss { 4 }
    else if d_end < p.horizon {
        let atr_hit = p.atr_k.map_or(false, |k| s_end < hwm - k * p.atr_value);
        if atr_hit { 6 } else if pnl_end >= 0.0 { 5 } else { 3 }
    }
    else if pnl_end >= p.pnl_big { 0 }
    else if pnl_end >= 0.0 { 1 }
    else { 2 }
}
#[derive(Clone, Copy, PartialEq)]
enum OutcomeKind { StopLadder, GenericPnl, HoldNoStop }
impl OutcomeKind {
    fn levels(&self) -> Vec<&'static str> {
        match self {
            OutcomeKind::StopLadder => vec!["上漲解套", "橫盤震盪", "時間停損", "一般跌破停損", "黑天鵝熔斷停損"],
            OutcomeKind::GenericPnl => vec!["大幅獲利", "小幅獲利", "小幅虧損", "大幅虧損"],
            OutcomeKind::HoldNoStop => vec!["大幅獲利", "小幅獲利", "到期小幅虧損", "時間停損", "虧損熔斷", "區間止盈", "ATR停損"],
        }
    }
    fn stop_levels(&self) -> Vec<usize> {
        match self { OutcomeKind::StopLadder => vec![3, 4], OutcomeKind::GenericPnl => vec![3], OutcomeKind::HoldNoStop => vec![4] }
    }
    fn style(&self, i: usize) -> (&'static str, &'static str) {
        match self {
            OutcomeKind::StopLadder => [("上漲典型 (Bullish)", "green"), ("橫盤典型", "blue"), ("時間停損典型", "purple"),
                                        ("跌破停損典型", "red"), ("黑天鵝典型 (跳空穿越)", "black")][i],
            OutcomeKind::GenericPnl => [("大幅獲利典型", "green"), ("小幅獲利典型", "blue"),
                                        ("小幅虧損典型", "purple"), ("大幅虧損典型", "red")][i],
            OutcomeKind::HoldNoStop => [("大幅獲利典型", "green"), ("小幅獲利典型", "blue"), ("到期虧損典型", "purple"),
                                        ("時間停損典型", "darkorange"), ("虧損熔斷典型", "crimson"), ("區間止盈典型", "teal"), ("ATR停損典型", "#c026d3")][i],
        }
    }
    fn classify(&self, r: &Res, cfg: &Cfg) -> usize {
        let n = r.s.len();
        let (s_end, pnl_end, d_end) = (r.s[n - 1], r.pnl[n - 1], r.dates[n - 1]);
        match self {
            OutcomeKind::StopLadder => {
                if s_end < cfg.swan_price || pnl_end < cfg.swan_pnl0 - cfg.swan_extra_loss { 4 }
                else if s_end < cfg.stop_price && d_end < cfg.horizon { 3 }
                else if s_end >= cfg.bull_price { 0 }
                else if d_end < cfg.horizon { 2 }
                else { 1 }
            }
            OutcomeKind::HoldNoStop => {
                let hwm = r.s.iter().cloned().fold(f64::MIN, f64::max);
                let hp = HoldParams { swan_pnl0: cfg.swan_pnl0, swan_extra_loss: cfg.swan_extra_loss, horizon: cfg.horizon,
                                      pnl_big: cfg.pnl_big, atr_k: cfg.atr_k, atr_value: cfg.atr_value };
                classify_hold(&hp, s_end, pnl_end, d_end, hwm)
            }
            OutcomeKind::GenericPnl => {
                let big = cfg.pnl_big;
                if pnl_end >= big { 0 } else if pnl_end >= 0.0 { 1 } else if pnl_end > -big { 2 } else { 3 }
            }
        }
    }
}

#[derive(Clone)]
struct Ctx {
    label: String,
    spec: String,
    debit: f64,
    spot: f64,
    iv_init: Vec<f64>,
    iv_hist: Vec<(Day, Vec<f64>)>, // 每次更新週期的 IV（生效日, 各腿 IV）；模擬用最新收盤日當時生效的那組
    today: Day,
    sim_start: Day,
    short_term_date: Day,
    pricer: PricerKind,
    env: Env,
    path_model: PathKind,
    vol: f64,
    drift: f64,
    n_paths: usize,
    sv: Sv,
    seed: SeedFn,
    oos_seed: SeedFn,
    oos_n: usize,
    cfg: Cfg,
    outcome: OutcomeKind,
    exit_spec: RuleSpec,
    trendlines: Vec<Trendline>,
    market_year: i32,
    market: String,
    stop_variants: Vec<RuleSpec>,
    sens_rules: Vec<RuleSpec>,
    auto_tp: Option<AutoTp>,
    tp_variants: Vec<RuleSpec>,
    drift_rules: Vec<RuleSpec>,
    sens_n: usize,
    sens_scenarios: Vec<Scenario>,
    drift_grid: Vec<f64>,
    drift_n: usize,
    reports: Vec<&'static str>,
    heavy_reports: Vec<&'static str>,
    take_profit_line: Option<f64>,
    daily_grid: bool,
    actual: Vec<(Day, f64)>, // 已發生的實際收盤（從模擬起點依序對應 dates 的前幾格）
    now: Day,                // 最新實際收盤日（模擬由此之後開始）
    match_days: usize,       // 選獲利代表路徑時，比對最新實際收盤之後前幾個交易日
    // 由 setup_trade 填入
    strategy: Strategy,
    s0: f64,
    dates: Vec<Day>,
}

// ---- 定價 ---------------------------------------------------------------------------
fn pnorm(x: f64) -> f64 {
    let xabs = x.abs();
    let c;
    if xabs > 37.0 {
        c = 0.0;
    } else {
        let e = (-xabs * xabs / 2.0).exp();
        if xabs < 7.07106781186547 {
            let mut b = 3.52624965998911E-02 * xabs + 0.700383064443688;
            b = b * xabs + 6.37396220353165;
            b = b * xabs + 33.912866078383;
            b = b * xabs + 112.079291497871;
            b = b * xabs + 221.213596169931;
            b = b * xabs + 220.206867912376;
            let num = e * b;
            let mut d = 8.83883476483184E-02 * xabs + 1.75566716318264;
            d = d * xabs + 16.064177579207;
            d = d * xabs + 86.7807322029461;
            d = d * xabs + 296.564248779674;
            d = d * xabs + 637.333633378831;
            d = d * xabs + 793.826512519948;
            d = d * xabs + 440.413735824752;
            c = num / d;
        } else {
            let mut b = xabs + 0.65;
            b = xabs + 4.0 / b;
            b = xabs + 3.0 / b;
            b = xabs + 2.0 / b;
            b = xabs + 1.0 / b;
            c = e / b / 2.506628274631;
        }
    }
    if x > 0.0 { 1.0 - c } else { c }
}
fn bs_price(s: f64, k: f64, t: f64, is_call: bool, r: f64, sig: f64) -> f64 {
    let d1 = ((s / k).ln() + (r + sig * sig / 2.0) * t) / (sig * t.sqrt());
    let d2 = d1 - sig * t.sqrt();
    if is_call { s * pnorm(d1) - k * (-r * t).exp() * pnorm(d2) }
    else { k * (-r * t).exp() * pnorm(-d2) - s * pnorm(-d1) }
}
fn intrinsic(s: f64, k: f64, is_call: bool) -> f64 {
    if is_call { (s - k).max(0.0) } else { (k - s).max(0.0) }
}
fn price(kind: PricerKind, s: f64, k: f64, t: f64, is_call: bool, env: &Env) -> f64 {
    if t <= 0.0 { return intrinsic(s, k, is_call); }
    match kind {
        PricerKind::Bs => {
            let base = env.sigma;
            let sig = (base * (1.0 - env.skew * (s / env.s0_ref - 1.0))).max(0.25 * base);
            bs_price(s, k, t, is_call, env.r, sig)
        }
        PricerKind::BsDyn => {
            let base = env.sigma * env.vf;
            let sk = env.skew * (1.0 + env.term_a * (-t / env.term_tau).exp());
            let sig = (base * (1.0 - sk * (s / env.s0_ref - 1.0))).max(0.25 * base);
            bs_price(s, k, t, is_call, env.r, sig)
        }
    }
}

fn leg_value(leg: &Leg, s: f64, d: Day, pk: PricerKind, env: &Env, s_exp: Option<f64>) -> f64 {
    let (mut s, mut d) = (s, d);
    if d > leg.expiry {
        if let Some(x) = s_exp { s = x; }
        d = leg.expiry;
    }
    let t = ((leg.expiry - d) as f64 / 365.0).max(0.0);
    let mut e = *env;
    if !leg.iv.is_nan() { e.sigma = leg.iv; }
    price(pk, s, leg.k, t, leg.is_call, &e) * leg.qty * 100.0
}
fn calc_pnl(st: &Strategy, s: f64, d: Day, pk: PricerKind, env: &Env, s_exp: Option<&[f64]>) -> f64 {
    let v: f64 = st.legs.iter().enumerate()
        .map(|(i, l)| leg_value(l, s, d, pk, env, s_exp.map(|m| m[i]))).sum();
    v - st.debit
}

fn interp(xs: &[Day], ys: &[f64], x: Day) -> f64 {
    if x <= xs[0] { return ys[0]; }
    let n = xs.len();
    if x >= xs[n - 1] { return ys[n - 1]; }
    let j = xs.partition_point(|&v| v <= x);
    let (x0, x1) = (xs[j - 1], xs[j]);
    ys[j - 1] + (ys[j] - ys[j - 1]) * (x - x0) as f64 / (x1 - x0) as f64
}

// 每條路徑只定價一次；出場規則另外找出場點
fn path_pnl(st: &Strategy, pk: PricerKind, p: &Path, env: &Env) -> Vec<f64> {
    let s_exp: Vec<f64> = st.legs.iter().map(|l| interp(&p.dates, &p.s, l.expiry)).collect();
    let mut e = *env;
    (0..p.s.len()).map(|i| {
        if let Some(vf) = &p.vf { e.vf = vf[i]; }
        calc_pnl(st, p.s[i], p.dates[i], pk, &e, Some(&s_exp))
    }).collect()
}
fn exit_index(p: &Path, pnl: &[f64], ex: &ExitFn) -> usize {
    let n = p.s.len();
    let mut hwm = f64::MIN;
    for i in 0..n {
        hwm = hwm.max(p.s[i]);
        if ex(&State { date: p.dates[i], s: p.s[i], pnl: pnl[i], hwm }) { return i; }
    }
    n - 1
}

fn pnl_grid(st: &Strategy, pk: PricerKind, env: &Env, dates: &[Day], prices: &[f64]) -> Vec<Vec<f64>> {
    prices.iter().map(|&p| dates.iter().map(|&d| calc_pnl(st, p, d, pk, env, None)).collect()).collect()
}

// ---- 路徑模型 ---------------------------------------------------------------------------
// 前幾格填入實際收盤（固定不變），回傳模擬起始索引；沒有實際資料時沿用開盤跳空起點。
fn init_actual(ctx: &Ctx, s: &mut Vec<f64>, rng: &mut Rng) -> usize {
    if ctx.actual.is_empty() {
        s[0] = ctx.s0 * (1.0 + rng.norm() * ctx.cfg.open_shock_sd);
        return 1;
    }
    for (i, (d, p)) in ctx.actual.iter().enumerate() {
        assert_eq!(ctx.dates[i], *d, "實際收盤日期必須是模擬格點的前幾格");
        s[i] = *p;
    }
    ctx.actual.len().max(1)
}
fn gen_normal(seed: u64, drift: f64, vol: f64, ctx: &Ctx) -> Path {
    let cf = &ctx.cfg;
    let mut rng = Rng::new(seed);
    let n = ctx.dates.len();
    let mut s = vec![0.0; n];
    let k0 = init_actual(ctx, &mut s, &mut rng);
    let dz: Vec<f64> = (0..n - 1).map(|_| rng.norm()).collect();
    for i in k0..n {
        let dt = (ctx.dates[i] - ctx.dates[i - 1]) as f64 / 365.0;
        let p_jump = 1.0 - (-cf.jump_lambda * dt).exp();
        let shock = if rng.unif() < p_jump { cf.jump_size } else { 0.0 };
        s[i] = s[i - 1] * (1.0 + shock) * ((drift - 0.5 * vol * vol) * dt + vol * dt.sqrt() * dz[i - 1]).exp();
    }
    Path { dates: ctx.dates.clone(), s, vf: None }
}
fn gen_sv(seed: u64, drift: f64, vol: f64, ctx: &Ctx) -> Path {
    let cf = &ctx.cfg;
    let (kappa, xi, rho) = (ctx.sv.kappa, ctx.sv.xi, ctx.sv.rho);
    let mu_x = -0.5 * xi * xi / (2.0 * kappa);
    let mut rng = Rng::new(seed);
    let n = ctx.dates.len();
    let mut s = vec![0.0; n];
    let mut x = vec![0.0f64; n];
    let k0 = init_actual(ctx, &mut s, &mut rng);
    let z1: Vec<f64> = (0..n - 1).map(|_| rng.norm()).collect();
    let z2: Vec<f64> = (0..n - 1).map(|_| rng.norm()).collect();
    for i in k0..n {
        let dt = (ctx.dates[i] - ctx.dates[i - 1]) as f64 / 365.0;
        let v_t = vol * x[i - 1].exp();
        let jump = rng.unif() < 1.0 - (-cf.jump_lambda * dt).exp();
        let swan = if jump { cf.jump_size } else { 0.0 };
        s[i] = s[i - 1] * (1.0 + swan) * ((drift - 0.5 * v_t * v_t) * dt + v_t * dt.sqrt() * z1[i - 1]).exp();
        let zv = rho * z1[i - 1] + (1.0 - rho * rho).sqrt() * z2[i - 1];
        x[i] = x[i - 1] + kappa * (mu_x - x[i - 1]) * dt + xi * dt.sqrt() * zv + if jump { 0.3 } else { 0.0 };
    }
    let vf: Vec<f64> = x.iter().map(|v| v.exp()).collect();
    Path { dates: ctx.dates.clone(), s, vf: Some(vf) }
}
fn gen_path(i: usize, drift: f64, ctx: &Ctx, seed: SeedFn, kind: PathKind) -> Path {
    let sd = seed.base + (i as u64) * seed.mult; // i 為 1 起算，與 R 的 seed_fn(i) 一致
    match kind {
        PathKind::Normal => gen_normal(sd, drift, ctx.vol, ctx),
        PathKind::Sv => gen_sv(sd, drift, ctx.vol, ctx),
    }
}
fn build_paths(n: usize, drift: f64, ctx: &Ctx, seed: SeedFn, kind: PathKind) -> Vec<Path> {
    par_map(n, |i| gen_path(i + 1, drift, ctx, seed, kind))
}
fn full_paths(results: &[Res], ctx: &Ctx) -> Vec<Path> {
    build_paths(results.len(), ctx.drift, ctx, ctx.seed, ctx.path_model)
}

// ---- 出場規則建構 ---------------------------------------------------------------------------
// 階梯式出場規則的參數與判斷（純函數，沒有 Ctx / 字串 / 集合，方便形式化驗證與單元測試）
#[derive(Clone, Copy)]
struct LadderParams {
    horizon: Day, stop: Option<f64>, ts: bool, ts_date: Day, br: bool, wtp: Option<f64>,
    extra: f64, pnl0: f64, swan_px: f64, atr_d: Option<f64>,
}
#[derive(Clone, Copy, PartialEq, Debug)]
enum ExitReason { PriceStop, Breaker, Atr, TimeStop, WindowTp, Expiry }
impl ExitReason {
    fn label(self) -> &'static str {
        match self {
            ExitReason::PriceStop => "價格停損", ExitReason::Breaker => "虧損熔斷", ExitReason::Atr => "ATR停損",
            ExitReason::TimeStop => "時間停損", ExitReason::WindowTp => "區間止盈", ExitReason::Expiry => "到期",
        }
    }
}
fn ladder_params(spec: &RuleSpec, ctx: &Ctx) -> LadderParams {
    let cf = &ctx.cfg;
    let stop = match spec.stop {
        StopArg::Default => Some(cf.stop_price),
        StopArg::Price(p) => Some(p),
        StopArg::None => None,
    };
    LadderParams {
        horizon: cf.horizon, stop, ts: spec.time_stop, ts_date: cf.time_stop_date, br: spec.breaker, wtp: spec.window_tp,
        extra: cf.swan_extra_loss, pnl0: cf.swan_pnl0,
        // 熔斷價格 = 停損價 - (stop_price - swan_price)；沒有價格停損時只留「虧損超過 swan_extra_loss」
        swan_px: stop.map_or(0.0, |s| s - (cf.stop_price - cf.swan_price)),
        atr_d: spec.atr.map(|k| k * cf.atr_value),
    }
}
// 回傳第一個成立的出場理由（優先順序：價格停損 > 熔斷 > ATR 停損 > 時間停損 > 區間止盈 > 到期）；都不成立回傳 None
fn ladder_reason(p: &LadderParams, d: Day, s: f64, pnl: f64, hwm: f64) -> Option<ExitReason> {
    if p.stop.map_or(false, |x| s < x) { return Some(ExitReason::PriceStop); }
    if p.br && ((pnl - p.pnl0) < -p.extra || s < p.swan_px) { return Some(ExitReason::Breaker); }
    if p.atr_d.map_or(false, |a| s < hwm - a) { return Some(ExitReason::Atr); }
    if p.ts && d >= p.ts_date && pnl < 0.0 { return Some(ExitReason::TimeStop); }
    if p.wtp.map_or(false, |cap| d >= p.ts_date && pnl >= 0.0 && pnl <= cap) { return Some(ExitReason::WindowTp); }
    if d >= p.horizon { return Some(ExitReason::Expiry); }
    None
}
fn ladder(spec: &RuleSpec, ctx: &Ctx) -> ExitFn {
    let p = ladder_params(spec, ctx);
    Box::new(move |s: &State| ladder_reason(&p, s.date, s.s, s.pnl, s.hwm).is_some())
}
// 出場規則只在最新實際收盤日之後才生效（已發生的日子不能回頭出場）
fn build_exit(spec: &RuleSpec, ctx: &Ctx) -> ExitFn {
    let inner = build_exit_inner(spec, ctx);
    let la = if ctx.actual.is_empty() { Day::MIN } else { ctx.now };
    Box::new(move |s: &State| s.date > la && inner(s))
}
fn build_exit_inner(spec: &RuleSpec, ctx: &Ctx) -> ExitFn {
    match &spec.kind {
        Kind::Ladder => ladder(spec, ctx),
        Kind::TakeProfit(t) => {
            let base = ladder(spec, ctx);
            let t = *t;
            Box::new(move |s: &State| base(s) || s.pnl >= t)
        }
        Kind::AnyOf(specs) => {
            let v: Vec<ExitFn> = specs.iter().map(|s| build_exit(s, ctx)).collect();
            Box::new(move |s: &State| v.iter().any(|r| r(s)))
        }
    }
}

struct RuleStats { pnl: Vec<f64>, early: Vec<bool>, days: Vec<f64> }
fn eval_rules(full: &[Path], exits: &[ExitFn], ctx: &Ctx) -> Vec<RuleStats> {
    let horizon = ctx.cfg.horizon;
    let per_path: Vec<Vec<(f64, bool, f64)>> = par_map(full.len(), |k| {
        let p = &full[k];
        let pnl = path_pnl(&ctx.strategy, ctx.pricer, p, &ctx.env);
        exits.iter().map(|ex| {
            let i = exit_index(p, &pnl, ex);
            (pnl[i], p.dates[i] < horizon, (p.dates[i] - p.dates[0]) as f64)
        }).collect()
    });
    (0..exits.len()).map(|j| RuleStats {
        pnl: per_path.iter().map(|x| x[j].0).collect(),
        early: per_path.iter().map(|x| x[j].1).collect(),
        days: per_path.iter().map(|x| x[j].2).collect(),
    }).collect()
}

fn classify_all(results: &[Res], ctx: &Ctx) -> Vec<usize> {
    results.iter().map(|r| ctx.outcome.classify(r, &ctx.cfg)).collect()
}

// ---- 輸出型別與 run -----------------------------------------------------------------------
enum Output { Table(Table), Html(String), Text(String) }
struct ReportOut { out: Output, publish_tp: Option<f64> }
fn tbl(t: Table) -> ReportOut { ReportOut { out: Output::Table(t), publish_tp: None } }

struct ReportDef {
    name: &'static str,
    title: &'static str,
    requires: fn(&Ctx) -> Result<(), &'static str>, // Err(缺少的欄位名)
    run: fn(&[Res], &Ctx) -> ReportOut,
}

fn run(ctx: &mut Ctx, report_names: &[&str]) -> (Vec<Res>, Vec<(&'static str, &'static str, ReportOut)>) {
    let ex = build_exit(&ctx.exit_spec, ctx);
    let results: Vec<Res> = par_map(ctx.n_paths, |i| {
        let p = gen_path(i + 1, ctx.drift, ctx, ctx.seed, ctx.path_model);
        let pnl = path_pnl(&ctx.strategy, ctx.pricer, &p, &ctx.env);
        let k = exit_index(&p, &pnl, &ex);
        Res { dates: p.dates[..=k].to_vec(), s: p.s[..=k].to_vec(), pnl: pnl[..=k].to_vec() }
    });
    let defs = reports();
    let mut out = Vec::new();
    for r in report_names {
        let d = match defs.iter().find(|d| d.name == *r) {
            Some(d) => d,
            None => panic!("未註冊報告: {}", r),
        };
        if let Err(miss) = (d.requires)(ctx) {
            eprintln!("略過報告 {}：缺少 {}", r, miss);
            continue;
        }
        let res = (d.run)(&results, ctx);
        if let Some(tp) = res.publish_tp { ctx.take_profit_line = Some(tp); }
        out.push((d.name, d.title, res));
    }
    (results, out)
}

// ---- 文字解析與對帳 ---------------------------------------------------------------------------
fn strategy_from_text(spec: &str, debit: f64, iv: &[f64]) -> Strategy {
    let legs = spec.split(';').enumerate().map(|(k, part)| {
        let f: Vec<&str> = part.split_whitespace().collect();
        let qty: f64 = f[0].parse().unwrap();
        let is_call = f[1].ends_with('C');
        let kk: f64 = f[1][..f[1].len() - 1].parse().unwrap();
        Leg { qty, k: kk, is_call, expiry: date(f[2]), iv: iv.get(k).copied().unwrap_or(f64::NAN) }
    }).collect();
    Strategy { debit, legs }
}

struct Market { prices: Vec<f64>, dates: Vec<Day>, z: Vec<Vec<f64>> }
fn market_from_text(txt: &str, year: i32) -> Market {
    let mut lines = txt.lines().map(|l| l.trim()).filter(|l| !l.is_empty());
    let header: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
    let dates: Vec<Day> = header[1..].iter().map(|h| {
        let p: Vec<&str> = h.split('/').collect();
        days_from_civil(year, p[0].parse().unwrap(), p[1].parse().unwrap())
    }).collect();
    let (mut prices, mut z) = (Vec::new(), Vec::new());
    for l in lines {
        let v: Vec<f64> = l.split_whitespace().map(|x| x.parse().unwrap()).collect();
        prices.push(v[0]);
        z.push(v[1..].to_vec());
    }
    Market { prices, dates, z }
}

fn validate_vs_broker(st: &Strategy, pk: PricerKind, env: &Env, mk: &Market) {
    let zm = pnl_grid(st, pk, env, &mk.dates, &mk.prices);
    let (mut mx, mut sum, mut bad, mut cnt) = (0.0f64, 0.0, 0usize, 0usize);
    let mut cells: Vec<(f64, usize, usize)> = Vec::new();
    for i in 0..mk.prices.len() {
        for j in 0..mk.dates.len() {
            let d = zm[i][j] - mk.z[i][j];
            let tol = if mk.z[i][j].abs() >= 9990.0 { 150.0 } else { 30.0 };
            mx = mx.max(d.abs());
            sum += d.abs();
            cnt += 1;
            if d.abs() > tol { bad += 1; }
            cells.push((d, i, j));
        }
    }
    println!("對帳：最大誤差 ${:.0} | 平均誤差 ${:.0} | 超出容差格數 {} / {}", mx, sum / cnt as f64, bad, cnt);
    cells.sort_by(|a, b| b.0.abs().partial_cmp(&a.0.abs()).unwrap());
    let top: Vec<String> = cells.iter().take(5).map(|(d, i, j)|
        format!("${} @{}：{:+.0}", mk.prices[*i], fmt_md(mk.dates[*j]), d)).collect();
    println!("誤差最大的格子（價格 @ 日期：模型 - 券商）：{}", top.join("；"));
}

fn nelder_mead(f: &dyn Fn(&[f64]) -> f64, start: &[f64], lo: f64, hi: f64) -> (Vec<f64>, f64) {
    let n = start.len();
    let clamp = |v: &mut Vec<f64>| for x in v.iter_mut() { *x = x.clamp(lo, hi); };
    let mut pts: Vec<Vec<f64>> = vec![start.to_vec()];
    for i in 0..n {
        let mut p = start.to_vec();
        p[i] += 0.05;
        clamp(&mut p);
        pts.push(p);
    }
    let mut vals: Vec<f64> = pts.iter().map(|p| f(p)).collect();
    for _ in 0..600 {
        let mut idx: Vec<usize> = (0..=n).collect();
        idx.sort_by(|&a, &b| vals[a].partial_cmp(&vals[b]).unwrap());
        pts = idx.iter().map(|&i| pts[i].clone()).collect();
        vals = idx.iter().map(|&i| vals[i]).collect();
        if (vals[n] - vals[0]).abs() < 1e-9 { break; }
        let cen: Vec<f64> = (0..n).map(|d| pts[..n].iter().map(|p| p[d]).sum::<f64>() / n as f64).collect();
        let mk = |t: f64| { let mut v: Vec<f64> = (0..n).map(|d| cen[d] + t * (pts[n][d] - cen[d])).collect(); clamp(&mut v); v };
        let xr = mk(-1.0);
        let fr = f(&xr);
        if fr < vals[0] {
            let xe = mk(-2.0);
            let fe = f(&xe);
            if fe < fr { pts[n] = xe; vals[n] = fe; } else { pts[n] = xr; vals[n] = fr; }
        } else if fr < vals[n - 1] {
            pts[n] = xr; vals[n] = fr;
        } else {
            let (xc, fc) = if fr < vals[n] { let x = mk(-0.5); let v = f(&x); (x, v) } else { let x = mk(0.5); let v = f(&x); (x, v) };
            if fc < vals[n].min(fr) { pts[n] = xc; vals[n] = fc; }
            else {
                let best = pts[0].clone();
                for i in 1..=n {
                    let mut v: Vec<f64> = (0..n).map(|d| best[d] + 0.5 * (pts[i][d] - best[d])).collect();
                    clamp(&mut v);
                    vals[i] = f(&v);
                    pts[i] = v;
                }
            }
        }
    }
    let b = (0..=n).min_by(|&a, &c| vals[a].partial_cmp(&vals[c]).unwrap()).unwrap();
    (pts[b].clone(), vals[b])
}

fn calibrate_iv(st: &Strategy, pk: PricerKind, env: &Env, mk: &Market, start: &[f64]) -> (Vec<f64>, f64, f64) {
    let obj = |iv: &[f64]| {
        let mut s2 = st.clone();
        for (k, l) in s2.legs.iter_mut().enumerate() { l.iv = iv[k]; }
        let zm = pnl_grid(&s2, pk, env, &mk.dates, &mk.prices);
        let (mut se, mut c) = (0.0, 0usize);
        for i in 0..mk.prices.len() {
            for j in 0..mk.dates.len() {
                if mk.z[i][j].abs() < 9990.0 { let d = zm[i][j] - mk.z[i][j]; se += d * d; c += 1; }
            }
        }
        (se / c as f64).sqrt()
    };
    let (iv, rmse) = nelder_mead(&obj, start, 0.2, 2.5);
    let r0 = obj(start);
    (iv, rmse, r0)
}

// 由 trade() 組出 ctx：對帳、IV 校準、模擬日期、今日損益
fn setup_trade(ctx: &mut Ctx) {
    let mut st = strategy_from_text(&ctx.spec, ctx.debit, &ctx.iv_init);
    let mk = if ctx.market.trim().is_empty() { None } else { Some(market_from_text(&ctx.market, ctx.market_year)) };
    if let Some(mk) = &mk {
        let env_flat = Env { skew: 0.0, vf: 1.0, term_a: 0.0, ..ctx.env };
        validate_vs_broker(&st, ctx.pricer, &env_flat, mk);
        let (iv, rmse, r0) = calibrate_iv(&st, ctx.pricer, &env_flat, mk, &ctx.iv_init);
        for (k, l) in st.legs.iter_mut().enumerate() { l.iv = iv[k]; }
        let s: Vec<String> = iv.iter().map(|v| format!("{:.1}%", v * 100.0)).collect();
        if !quiet() { println!("校準 IV：{}（RMSE {:.1}，起始 RMSE {:.1}）", s.join(" / "), rmse, r0); }
    } else {
        let s: Vec<String> = st.legs.iter().map(|l| format!("{:.1}%", l.iv * 100.0)).collect();
        if !quiet() { println!("IV 直接採用實測（依腿順序）：{}", s.join(" / ")); }
    }
    let horizon = st.legs.iter().map(|l| l.expiry).min().unwrap();
    ctx.cfg.horizon = horizon;
    ctx.cfg.pnl_today = calc_pnl(&st, ctx.spot, ctx.today, ctx.pricer, &ctx.env, None);
    ctx.cfg.swan_pnl0 = if ctx.cfg.swan_loss_from_today { ctx.cfg.pnl_today } else { 0.0 };

    let mdates: Vec<Day> = mk.as_ref().map_or(vec![], |m|
        m.dates.iter().copied().filter(|&d| d >= ctx.today && d <= horizon + 1).collect());
    let mut ds: Vec<Day> = vec![ctx.today, horizon];
    ds.extend(mdates);
    ds.extend((0..7).map(|k| ctx.sim_start + k));
    if ctx.daily_grid {
        let mut d = ctx.sim_start;
        while d <= horizon {
            if iso_weekday(d) <= 5 { ds.push(d); }
            d += 1;
        }
    }
    ds.sort();
    ds.dedup();
    ds.retain(|&d| iso_weekday(d) <= 5); // 只保留交易日（週一到週五）
    ctx.dates = ds;
    ctx.strategy = st;
    if let Some(&(d, p)) = ctx.actual.last() { ctx.now = d; ctx.s0 = p; } else { ctx.now = ctx.today; ctx.s0 = ctx.spot; }
    for (i, (d, _)) in ctx.actual.iter().enumerate() {
        if ctx.dates.get(i) != Some(d) { panic!("實際收盤日期 {} 不在模擬格點前段", fmt_md(*d)); }
    }
    if !quiet() {
        for &(d, p) in &ctx.actual {
            let v = calc_pnl(&strategy_at(ctx, d), p, d, ctx.pricer, &ctx.env, None);
            println!("模型估值：{} 實際收盤 ${:.2} 時，部位損益約 ${:.0}（用當時生效的 IV，請與券商現值對照）", fmt_md(d), p, v);
        }
    }
}

// ---- 每次更新週期的 IV：取「生效日 <= d」中最新的一組；早於第一組就用第一組 ----------------------------
fn iv_at(hist: &[(Day, Vec<f64>)], d: Day) -> Vec<f64> {
    hist.iter().filter(|h| h.0 <= d).last().or(hist.first()).map(|h| h.1.clone()).unwrap_or_default()
}
fn strategy_at(ctx: &Ctx, d: Day) -> Strategy {
    let mut st = ctx.strategy.clone();
    if !ctx.iv_hist.is_empty() {
        let iv = iv_at(&ctx.iv_hist, d);
        for (k, l) in st.legs.iter_mut().enumerate() { if let Some(&v) = iv.get(k) { l.iv = v; } }
    }
    st
}
fn price_for_pnl(st: &Strategy, ctx: &Ctx, d: Day, target: f64) -> f64 {
    let (mut lo, mut hi) = (1.0, 1000.0);
    for _ in 0..80 {
        let mid = (lo + hi) / 2.0;
        if calc_pnl(st, mid, d, ctx.pricer, &ctx.env, None) < target { lo = mid } else { hi = mid }
    }
    (lo + hi) / 2.0
}
// 依現行出場規則，判斷某個實際收盤日是否已經觸發出場（建倉日當天不算）
fn exit_reason(ctx: &Ctx, d: Day, s: f64, pnl: f64, hwm: f64) -> Option<&'static str> {
    ladder_reason(&ladder_params(&ctx.exit_spec, ctx), d, s, pnl, hwm).map(|r| r.label())
}
// 建倉日以來每個實際收盤日的狀態表；回傳第一個觸發出場的 (日期, 原因)
fn rep_actual_path(ctx: &Ctx, print: bool) -> Option<(Day, &'static str)> {
    let cf = &ctx.cfg;
    let mut rows = Vec::new();
    let mut first: Option<(Day, &'static str)> = None;
    let mut hwm = f64::MIN;
    for (i, &(d, p)) in ctx.actual.iter().enumerate() {
        hwm = hwm.max(p);
        let st = strategy_at(ctx, d);
        let pnl = calc_pnl(&st, p, d, ctx.pricer, &ctx.env, None);
        let br = price_for_pnl(&st, ctx, d, cf.swan_pnl0 - cf.swan_extra_loss);
        let be = price_for_pnl(&st, ctx, d, 0.0);
        let tp = ctx.exit_spec.window_tp.map_or(f64::NAN, |c| price_for_pnl(&st, ctx, d, c));
        let reason = if i >= 1 && first.is_none() { exit_reason(ctx, d, p, pnl, hwm) } else { None };
        if let Some(r) = reason { first = Some((d, r)); }
        let iv: Vec<String> = st.legs.iter().map(|l| format!("{:.0}%", l.iv * 100.0)).collect();
        rows.push(vec![fmt_md(d), f2(p), iv.join("/"), f1(pnl), f2(br), f2(be), f2(tp),
                       match (i, reason) { (0, _) => "建倉日".to_string(), (_, Some(r)) => format!("已觸發：{}", r), _ => "持有".to_string() }]);
    }
    if print {
        Table { headers: ["日期", "收盤", "IV(各腿)", "實際損益", "熔斷價", "兩平價", "止盈價", "狀態"].iter().map(|x| x.to_string()).collect(), rows }.print();
        if let Some((d, r)) = first {
            println!("注意：依現行規則，實際路徑已在 {} 觸發「{}」。以下模擬仍假設繼續持有，僅供參考。", fmt_md(d), r);
        }
    }
    first
}

// =============================================================================
// PART 2：報告模組（REPORTS 註冊表）
// =============================================================================
fn req_ok(_: &Ctx) -> Result<(), &'static str> { Ok(()) }
fn req_trend(c: &Ctx) -> Result<(), &'static str> { if c.trendlines.is_empty() { Err("trendlines") } else { Ok(()) } }
fn req_stop_var(c: &Ctx) -> Result<(), &'static str> { if c.stop_variants.is_empty() { Err("stop_variants") } else { Ok(()) } }
fn req_tp_var(c: &Ctx) -> Result<(), &'static str> { if c.tp_variants.is_empty() { Err("tp_variants") } else { Ok(()) } }
fn req_auto(c: &Ctx) -> Result<(), &'static str> { if c.auto_tp.is_none() { Err("auto_tp") } else { Ok(()) } }
fn req_sens(c: &Ctx) -> Result<(), &'static str> {
    if c.sens_scenarios.is_empty() || c.sens_rules.is_empty() { Err("sens_scenarios / sens_rules") } else { Ok(()) }
}
fn req_drift(c: &Ctx) -> Result<(), &'static str> {
    if c.drift_rules.is_empty() || c.drift_grid.is_empty() { Err("drift_rules / drift_grid") } else { Ok(()) }
}

fn reports() -> Vec<ReportDef> {
    vec![
        ReportDef { name: "three_day_up_probability", title: "短期走勢預測", requires: req_ok, run: rep_short_up },
        ReportDef { name: "outcome_probabilities", title: "情境機率", requires: req_ok, run: rep_outcome_prob },
        ReportDef { name: "expected_value", title: "蒙地卡羅期望值與極端風險分析", requires: req_ok, run: rep_expected_value },
        ReportDef { name: "boundary_expectations", title: "趨勢線與停損觸及期望值分析", requires: req_trend, run: rep_boundary },
        ReportDef { name: "post_breakout", title: "突破後前瞻期望值（扣除同日基準）", requires: req_trend, run: rep_post_breakout },
        ReportDef { name: "stop_rule_scan", title: "停損規則掃描", requires: req_stop_var, run: rep_stop_scan },
        ReportDef { name: "stop_rule_scan_oos", title: "停損規則掃描（樣本外，新隨機種子）", requires: req_stop_var, run: rep_stop_scan_oos },
        ReportDef { name: "take_profit_scan", title: "止盈門檻掃描（總損益達標即出場）", requires: req_tp_var, run: rep_tp_scan },
        ReportDef { name: "take_profit_scan_oos", title: "止盈門檻掃描（樣本外，新隨機種子）", requires: req_tp_var, run: rep_tp_scan_oos },
        ReportDef { name: "auto_take_profit", title: "自動決定止盈線（淨損益已扣平倉成本；選用門檻會畫在熱力圖上）", requires: req_auto, run: rep_auto_tp },
        ReportDef { name: "sensitivity_scenarios", title: "模型假設敏感度（已做起始估值校正，E[PnL] 水位可跨情境比較）", requires: req_sens, run: rep_sensitivity },
        ReportDef { name: "drift_scan", title: "不同 drift 下的停損與止盈規則比較（同一組路徑）", requires: req_drift, run: rep_drift_scan },
        ReportDef { name: "heatmap_typical_representative", title: "熱力圖", requires: req_ok, run: rep_heatmap },
        ReportDef { name: "breaker_envelope", title: "熔斷出場每日最低點（模擬的實際止損連線）", requires: req_ok, run: |r, c| rep_breaker_env(r, c) },
        ReportDef { name: "heatmap_ascii", title: "文字熱力圖（可在 Rust Playground 直接顯示）", requires: req_ok, run: rep_heatmap_ascii },
    ]
}

fn rep_short_up(results: &[Res], ctx: &Ctx) -> ReportOut {
    let up = results.iter().filter(|r|
        r.dates.iter().position(|&d| d == ctx.short_term_date).map_or(false, |i| r.s[i] > ctx.s0)).count();
    tbl(Table {
        headers: vec!["Metric".into(), "Probability_Pct".into()],
        rows: vec![vec![format!("至 {} 上漲機率（相對 ${:.2} 收盤）", fmt_md(ctx.short_term_date), ctx.s0),
                        f2(up as f64 / results.len() as f64 * 100.0)]],
    })
}

fn rep_outcome_prob(results: &[Res], ctx: &Ctx) -> ReportOut {
    let oc = classify_all(results, ctx);
    let lv = ctx.outcome.levels();
    let rows = lv.iter().enumerate().map(|(i, l)| {
        let c = oc.iter().filter(|&&x| x == i).count();
        vec![l.to_string(), f2(c as f64 / results.len() as f64 * 100.0)]
    }).collect();
    tbl(Table { headers: vec!["Outcome_Type".into(), "Probability_Pct".into()], rows })
}

fn rep_expected_value(results: &[Res], ctx: &Ctx) -> ReportOut {
    let fin: Vec<f64> = results.iter().map(|r| r.last_pnl()).collect();
    let n = fin.len() as f64;
    let se = sd(&fin) / n.sqrt();
    let m = mean(&fin);
    let win = fin.iter().filter(|&&x| x > 0.0).count() as f64 / n * 100.0;
    let mx = fin.iter().cloned().fold(f64::MIN, f64::max);
    let mn = fin.iter().cloned().fold(f64::MAX, f64::min);
    let mut rows: Vec<Vec<String>> = vec![
        ("期望值 E[PnL]", m), ("標準誤 SE", se), ("95% CI 下限", m - 1.96 * se), ("95% CI 上限", m + 1.96 * se),
        ("中位數", median(&fin)), ("標準差", sd(&fin)), ("勝率 P(PnL>0)", win),
        ("最大獲利", mx), ("最大虧損 (Tail Risk)", mn),
    ].into_iter().map(|(a, b)| vec![a.to_string(), f2(b)]).collect();
    let oc = classify_all(results, ctx);
    for (i, l) in ctx.outcome.levels().iter().enumerate() {
        let x: Vec<f64> = fin.iter().zip(&oc).filter(|&(_, &o)| o == i).map(|(v, _)| *v).collect();
        if x.is_empty() { continue; }
        rows.push(vec![format!("[{}] 條件期望值 (n={})", l, x.len()), f2(mean(&x))]);
    }
    tbl(Table { headers: vec!["Metric".into(), "Value".into()], rows })
}

fn rep_boundary(results: &[Res], ctx: &Ctx) -> ReportOut {
    let fin: Vec<f64> = results.iter().map(|r| r.last_pnl()).collect();
    let oc = classify_all(results, ctx);
    let stops = ctx.outcome.stop_levels();
    let mut sets: Vec<(String, Vec<bool>)> = ctx.trendlines.iter().map(|tl| {
        let hit = results.iter().map(|r| r.dates.iter().zip(&r.s).any(|(&d, &s)| {
            let v = trend_at(tl, d);
            v >= ctx.s0 && s >= v
        })).collect();
        (format!("向上觸及{}", tl.name), hit)
    }).collect();
    sets.push(("觸發停損機制".into(), oc.iter().map(|o| stops.contains(o)).collect()));
    let n = results.len() as f64;
    let rows = sets.iter().map(|(name, h)| {
        let sel: Vec<f64> = fin.iter().zip(h).filter(|&(_, &b)| b).map(|(v, _)| *v).collect();
        vec![name.clone(), sel.len().to_string(), f2(sel.len() as f64 / n * 100.0), f2(mean(&sel))]
    }).collect();
    tbl(Table { headers: ["觸及條件", "樣本數", "機率Pct", "期望值PnL"].iter().map(|s| s.to_string()).collect(), rows })
}

fn rep_post_breakout(results: &[Res], ctx: &Ctx) -> ReportOut {
    let fin: Vec<f64> = results.iter().map(|r| r.last_pnl()).collect();
    let oc = classify_all(results, ctx);
    let stops = ctx.outcome.stop_levels();
    let mut cache: HashMap<Day, f64> = HashMap::new();
    let mut rows = Vec::new();
    for tl in &ctx.trendlines {
        let (mut p_b, mut delta, mut base, mut stopped, mut bull) = (vec![], vec![], vec![], 0usize, 0usize);
        for (k, r) in results.iter().enumerate() {
            let brk = r.dates.iter().zip(&r.s).position(|(&d, &s)| { let v = trend_at(tl, d); v >= ctx.s0 && s >= v });
            if let Some(b) = brk {
                let d_b = r.dates[b];
                p_b.push(r.pnl[b]);
                delta.push(fin[k] - r.pnl[b]);
                let bd = *cache.entry(d_b).or_insert_with(|| {
                    let v: Vec<f64> = results.iter().map(|q| {
                        let j = q.dates.iter().rposition(|&x| x <= d_b).unwrap();
                        q.last_pnl() - q.pnl[j]
                    }).collect();
                    mean(&v)
                });
                base.push(bd);
                if stops.contains(&oc[k]) { stopped += 1; }
                if oc[k] == 0 { bull += 1; }
            }
        }
        let h = p_b.len();
        if h == 0 {
            rows.push(vec![tl.name.clone(), "0".into(), "NA".into(), "NA".into(), "NA".into(), "NA".into(), "NA".into(), "NA".into()]);
            continue;
        }
        let ex: Vec<f64> = delta.iter().zip(&base).map(|(d, b)| d - b).collect();
        rows.push(vec![tl.name.clone(), h.to_string(), f1(mean(&p_b)), f1(mean(&delta)), f1(mean(&base)),
                       f1(mean(&ex)), f1(stopped as f64 / h as f64 * 100.0), f1(bull as f64 / h as f64 * 100.0)]);
    }
    tbl(Table {
        headers: ["趨勢線", "突破樣本", "突破時PnL", "突破後增量", "同日基準增量", "超額增量", "突破後停損Pct", "突破後解套Pct"]
            .iter().map(|s| s.to_string()).collect(),
        rows,
    })
}

// ---- 規則掃描 -----------------------------------------------------------------------------------
const SCAN_COLS_SHORT: &[&str] = &["規則", "期望值", "標準誤", "相對現行Δ", "Δ標準誤", "t值", "標準差", "勝率Pct", "5%分位", "最大虧損", "最大獲利"];
const TP_COLS: &[&str] = &["規則", "期望值", "標準誤", "相對現行Δ", "Δ標準誤", "t值", "標準差", "勝率Pct", "5%分位", "最大虧損", "最大獲利", "提前出場Pct", "平均持有天數"];

fn stop_scan_rows(full: &[Path], ctx: &Ctx, variants: &[RuleSpec], cols: Option<&[&str]>) -> Table {
    let exits: Vec<ExitFn> = variants.iter().map(|v| build_exit(v, ctx)).collect();
    let stats = eval_rules(full, &exits, ctx);
    let base = &stats[0].pnl;
    let n = base.len() as f64;
    let all_names = ["規則", "期望值", "標準誤", "相對現行Δ", "Δ標準誤", "t值", "中位數", "標準差", "報酬風險比",
                     "勝率Pct", "5%分位", "最大虧損", "最大獲利", "提前出場Pct", "平均持有天數"];
    let rows_all: Vec<Vec<String>> = stats.iter().enumerate().map(|(j, s)| {
        let d: Vec<f64> = s.pnl.iter().zip(base).map(|(a, b)| a - b).collect();
        let dse = sd(&d) / n.sqrt();
        let tv = if dse > 1e-8 { f2(mean(&d) / dse) } else { "NA".into() };
        vec![
            variants[j].name.clone(), f1(mean(&s.pnl)), f1(sd(&s.pnl) / n.sqrt()), f1(mean(&d)), f1(dse), tv,
            f1(median(&s.pnl)), f1(sd(&s.pnl)), f3(mean(&s.pnl) / sd(&s.pnl)),
            f1(s.pnl.iter().filter(|&&x| x > 0.0).count() as f64 / n * 100.0),
            f1(quantile(&s.pnl, 0.05)),
            f1(s.pnl.iter().cloned().fold(f64::MAX, f64::min)),
            f1(s.pnl.iter().cloned().fold(f64::MIN, f64::max)),
            f1(s.early.iter().filter(|&&x| x).count() as f64 / n * 100.0),
            f1(mean(&s.days)),
        ]
    }).collect();
    let sel: Vec<usize> = match cols {
        None => (0..all_names.len()).collect(),
        Some(c) => c.iter().map(|x| all_names.iter().position(|a| a == x).unwrap()).collect(),
    };
    Table {
        headers: sel.iter().map(|&i| all_names[i].to_string()).collect(),
        rows: rows_all.iter().map(|r| sel.iter().map(|&i| r[i].clone()).collect()).collect(),
    }
}

fn rep_stop_scan(results: &[Res], ctx: &Ctx) -> ReportOut {
    tbl(stop_scan_rows(&full_paths(results, ctx), ctx, &ctx.stop_variants, None))
}
fn rep_stop_scan_oos(_: &[Res], ctx: &Ctx) -> ReportOut {
    let p = build_paths(ctx.oos_n, ctx.drift, ctx, ctx.oos_seed, ctx.path_model);
    tbl(stop_scan_rows(&p, ctx, &ctx.stop_variants, None))
}
fn rep_tp_scan(results: &[Res], ctx: &Ctx) -> ReportOut {
    tbl(stop_scan_rows(&full_paths(results, ctx), ctx, &ctx.tp_variants, Some(TP_COLS)))
}
fn rep_tp_scan_oos(_: &[Res], ctx: &Ctx) -> ReportOut {
    let p = build_paths(ctx.oos_n, ctx.drift, ctx, ctx.oos_seed, ctx.path_model);
    tbl(stop_scan_rows(&p, ctx, &ctx.tp_variants, Some(TP_COLS)))
}

// 自動止盈線：一個標準誤規則 + 樣本外檢驗（見 R 版註解）
fn rep_auto_tp(results: &[Res], ctx: &Ctx) -> ReportOut {
    let a = ctx.auto_tp.as_ref().unwrap();
    let inh = |mut sp: RuleSpec| { sp.stop = ctx.exit_spec.stop.clone(); sp.time_stop = ctx.exit_spec.time_stop; sp.breaker = ctx.exit_spec.breaker; sp };
    let mut specs = vec![inh(RuleSpec::new("不止盈"))];
    for g in &a.grid { specs.push(inh(RuleSpec::tp(&format!("止盈+{}", g), *g))); }
    let exits: Vec<ExitFn> = specs.iter().map(|s| build_exit(s, ctx)).collect();
    let util = |x: f64| match a.objective {
        Objective::Mean => x,
        Objective::LossAverse => if x >= 0.0 { x } else { a.lambda * x },
    };
    let score = |paths: &[Path]| -> Vec<(Vec<f64>, Vec<f64>)> {
        eval_rules(paths, &exits, ctx).into_iter().map(|st| {
            let net: Vec<f64> = st.pnl.iter().zip(&st.early).map(|(p, &e)| p - if e { a.slippage } else { 0.0 }).collect();
            let u: Vec<f64> = net.iter().map(|&x| util(x)).collect();
            (net, u)
        }).collect()
    };
    let ins = score(&full_paths(results, ctx));
    let oos = score(&build_paths(ctx.oos_n, ctx.drift, ctx, ctx.oos_seed, ctx.path_model));
    let ng = a.grid.len();
    let mu: Vec<f64> = (0..ng).map(|j| mean(&ins[j + 1].1)).collect();
    let best = (0..ng).max_by(|&x, &y| mu[x].partial_cmp(&mu[y]).unwrap()).unwrap();
    let ok: Vec<bool> = (0..ng).map(|j| {
        let d: Vec<f64> = ins[best + 1].1.iter().zip(&ins[j + 1].1).map(|(x, y)| x - y).collect();
        j == best || mean(&d) <= a.se_k * sd(&d) / (d.len() as f64).sqrt()
    }).collect();
    let pick = match a.prefer {
        Prefer::Higher => (0..ng).rev().find(|&j| ok[j]).unwrap(),
        Prefer::Lower => (0..ng).find(|&j| ok[j]).unwrap(),
        Prefer::Best => best,
    };
    let summ = |x: &(Vec<f64>, Vec<f64>)| (mean(&x.0), mean(&x.1),
        x.0.iter().filter(|&&v| v > 0.0).count() as f64 / x.0.len() as f64 * 100.0, quantile(&x.0, 0.05));
    let rows = (0..specs.len()).map(|j| {
        let (a1, a2, a3, a4) = summ(&ins[j]);
        let (b1, b2, b3, b4) = summ(&oos[j]);
        let (okc, pk) = if j == 0 { ("", "") } else {
            (if ok[j - 1] { "是" } else { "" }, if j - 1 == pick { "<== 選用" } else { "" })
        };
        vec![specs[j].name.clone(), f1(a1), f1(a2), f1(a3), f1(a4), f1(b1), f1(b2), f1(b3), f1(b4), okc.into(), pk.into()]
    }).collect();
    ReportOut {
        out: Output::Table(Table {
            headers: ["規則", "樣本內淨期望值", "樣本內效用", "樣本內勝率Pct", "樣本內5%分位", "樣本外淨期望值", "樣本外效用",
                      "樣本外勝率Pct", "樣本外5%分位", "與最佳無法區分", "選用"].iter().map(|s| s.to_string()).collect(),
            rows,
        }),
        publish_tp: Some(a.grid[pick]),
    }
}

// 模型假設敏感度（含起始估值校正）
fn rep_sensitivity(_: &[Res], ctx: &Ctx) -> ReportOut {
    let pnl0_base = calc_pnl(&ctx.strategy, ctx.s0, ctx.now, ctx.pricer, &ctx.env, None);
    let mut headers: Vec<String> = vec!["情境".into(), "起始偏移".into()];
    let mut rows = Vec::new();
    for sc in &ctx.sens_scenarios {
        let mut c2 = ctx.clone();
        c2.env.skew = sc.skew.unwrap_or(ctx.env.skew);
        c2.env.term_a = sc.term_a;
        c2.env.term_tau = sc.term_tau;
        c2.vol = sc.vol.unwrap_or(ctx.vol);
        c2.pricer = sc.pricer;
        for l in c2.strategy.legs.iter_mut() { l.iv *= sc.iv_scale; }
        let offset = calc_pnl(&c2.strategy, c2.s0, c2.now, c2.pricer, &c2.env, None) - pnl0_base;
        c2.strategy.debit = ctx.strategy.debit + offset; // 起始估值校正
        let full = build_paths(ctx.sens_n, ctx.drift, &c2, ctx.seed, sc.path);
        let tb = stop_scan_rows(&full, &c2, &ctx.sens_rules, Some(SCAN_COLS_SHORT));
        if headers.len() == 2 { headers.extend(tb.headers.clone()); }
        for r in tb.rows {
            let mut row = vec![sc.name.clone(), f1(offset)];
            row.extend(r);
            rows.push(row);
        }
    }
    tbl(Table { headers, rows })
}

fn rep_drift_scan(results: &[Res], ctx: &Ctx) -> ReportOut {
    let n = ctx.drift_n.min(results.len());
    let mut headers: Vec<String> = vec!["年化drift".into()];
    let mut rows = Vec::new();
    for &mu in &ctx.drift_grid {
        let p = build_paths(n, mu, ctx, ctx.seed, ctx.path_model);
        let tb = stop_scan_rows(&p, ctx, &ctx.drift_rules, Some(TP_COLS));
        if headers.len() == 1 { headers.extend(tb.headers.clone()); }
        for r in tb.rows {
            let mut row = vec![format!("{}", mu)];
            row.extend(r);
            rows.push(row);
        }
    }
    tbl(Table { headers, rows })
}

// ---- 熱力圖（輸出獨立 HTML，使用 Plotly.js CDN）----------------------------------------------------
fn js_nums(v: &[f64]) -> String {
    let s: Vec<String> = v.iter().map(|x| if x.is_finite() { format!("{}", x) } else { "null".into() }).collect();
    format!("[{}]", s.join(","))
}
fn js_str(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}
fn js_strs(v: &[String]) -> String {
    format!("[{}]", v.iter().map(|s| js_str(s)).collect::<Vec<_>>().join(","))
}

fn medoid_key(keys: &[usize], results: &[Res], l: usize) -> usize {
    let rows: Vec<Vec<f64>> = keys.iter().map(|&k| {
        let s = &results[k].s;
        let mut v: Vec<f64> = s.iter().map(|x| x.ln()).collect();
        let last = *v.last().unwrap();
        v.resize(l, last);
        v
    }).collect();
    let center: Vec<f64> = (0..l).map(|j| rows.iter().map(|r| r[j]).sum::<f64>() / rows.len() as f64).collect();
    let dist: Vec<f64> = rows.iter().map(|r| r.iter().zip(&center).map(|(a, b)| (a - b) * (a - b)).sum()).collect();
    let b = (0..keys.len()).min_by(|&x, &y| dist[x].partial_cmp(&dist[y]).unwrap()).unwrap();
    keys[b]
}

// ---- 熔斷路徑每日最低點：每個交易日，所有「被停損／熔斷出場」的路徑中最低的股價 -------------------------
fn breaker_env(results: &[Res], ctx: &Ctx, oc: &[usize]) -> Vec<(Day, f64, f64, usize)> {
    let stops = ctx.outcome.stop_levels();
    let first = ctx.actual.len().saturating_sub(1); // 最新一個實際收盤的格點
    let mut out = Vec::new();
    for j in (first + 1)..ctx.dates.len() {
        // 在第 j 個格點被熔斷／停損出場的路徑（路徑長度 = j + 1）
        let px: Vec<f64> = results.iter().enumerate()
            .filter(|(k, r)| stops.contains(&oc[*k]) && r.s.len() == j + 1)
            .map(|(_, r)| r.s[j]).collect();
        if px.is_empty() { continue; }
        let mn = px.iter().cloned().fold(f64::MAX, f64::min);
        let mx = px.iter().cloned().fold(f64::MIN, f64::max);
        out.push((ctx.dates[j], mn, mx, px.len()));
    }
    out
}
// 理論熔斷價：該日使總損益剛好等於「熔斷門檻」的股價（損益只和股價、日期有關，二分法求解）
fn breaker_price(ctx: &Ctx, d: Day) -> f64 {
    let target = ctx.cfg.swan_pnl0 - ctx.cfg.swan_extra_loss;
    let (mut lo, mut hi) = (5.0, 400.0);
    for _ in 0..80 {
        let mid = (lo + hi) / 2.0;
        if calc_pnl(&ctx.strategy, mid, d, ctx.pricer, &ctx.env, None) < target { lo = mid } else { hi = mid }
    }
    (lo + hi) / 2.0
}
fn rep_breaker_env(results: &[Res], ctx: &Ctx) -> ReportOut {
    let oc = classify_all(results, ctx);
    let env = breaker_env(results, ctx, &oc);
    let rows = env.iter().map(|&(d, mn, mx, cnt)| {
        vec![fmt_md(d), cnt.to_string(), f2(mn), f2(mx), f2(breaker_price(ctx, d))]
    }).collect();
    tbl(Table {
        headers: ["日期", "當日熔斷出場數", "最低出場價", "最高出場價", "理論熔斷價"].iter().map(|x| x.to_string()).collect(),
        rows,
    })
}

// ---- 代表路徑 ---------------------------------------------------------------------------------
// 獲利路徑：在所有最終獲利的路徑中，選「模擬的前 match_days 個交易日最貼近最新實際收盤」的一條
//   （實際收盤那幾天對所有路徑都是固定的，無法比較，所以比的是實際收盤之後的前幾天：平均 |ln(S/最新收盤)| 最小）。
// 虧損路徑：依出場原因（各情境分類）各取一條最具代表性（medoid）的虧損路徑。
struct Rep { label: String, color: String, key: usize, cnt: usize }
fn pick_reps(results: &[Res], ctx: &Ctx, oc: &[usize]) -> Vec<Rep> {
    let l = ctx.dates.len();
    let first = ctx.actual.len().saturating_sub(1);
    let mut reps = Vec::new();
    let hz = ctx.cfg.horizon;
    let held = |r: &Res| *r.dates.last().unwrap() >= hz; // 持有到短腿到期
    let profit: Vec<usize> = (0..results.len()).filter(|&i| results[i].last_pnl() > 0.0 && held(&results[i])).collect();
    if !profit.is_empty() {
        let score = |i: usize| -> f64 {
            let r = &results[i];
            let end = (first + ctx.match_days).min(r.s.len() - 1);
            let v: Vec<f64> = ((first + 1)..=end).map(|j| (r.s[j] / ctx.s0).ln().abs()).collect();
            if v.is_empty() { 0.0 } else { mean(&v) }
        };
        let best = *profit.iter().min_by(|&&a, &&b| score(a).partial_cmp(&score(b)).unwrap()).unwrap();
        reps.push(Rep { label: format!("獲利路徑（前{}日最貼近實際收盤）", ctx.match_days), color: "green".into(), key: best, cnt: profit.len() });
    }
    for lv in 0..ctx.outcome.levels().len() {
        let keys: Vec<usize> = (0..results.len()).filter(|&i| oc[i] == lv && (results[i].last_pnl() <= 0.0 || !held(&results[i]))).collect();
        if keys.is_empty() { continue; }
        let (label, color) = ctx.outcome.style(lv);
        reps.push(Rep { label: label.into(), color: color.into(), key: medoid_key(&keys, results, l), cnt: keys.len() });
    }
    reps
}

fn rep_heatmap(results: &[Res], ctx: &Ctx) -> ReportOut {
    let show_stop = !matches!(ctx.exit_spec.stop, StopArg::None);
    let cf = &ctx.cfg;
    let gd = &ctx.dates;
    let lab: Vec<String> = gd.iter().map(|&d| fmt_md(d)).collect();
    let oc = classify_all(results, ctx);
    let first = ctx.actual.len().saturating_sub(1);
    let reps = pick_reps(results, ctx, &oc);
    let env = breaker_env(results, ctx, &oc);
    let mut rep_s: Vec<f64> = reps.iter().flat_map(|rp| results[rp.key].s.clone()).collect();
    rep_s.extend(env.iter().map(|e| e.1));
    let atr_lvl = ctx.exit_spec.atr.map(|k| ctx.actual.iter().map(|x| x.1).fold(f64::MIN, f64::max) - k * cf.atr_value);
    let mut y_lo = rep_s.iter().cloned().fold(f64::MAX, f64::min).min(cf.swan_price).min(cf.stop_price).min(ctx.s0).min(atr_lvl.unwrap_or(f64::MAX));
    let mut y_hi = rep_s.iter().cloned().fold(f64::MIN, f64::max).max(cf.stop_price).max(ctx.s0);
    for tl in &ctx.trendlines { y_hi = y_hi.max(trend_at(tl, gd[0])); }
    let pad = 0.05 * (y_hi - y_lo);
    y_lo = (y_lo - pad).floor();
    y_hi = (y_hi + pad).ceil();
    let np = 80;
    let prices: Vec<f64> = (0..np).map(|i| ((y_lo + (y_hi - y_lo) * i as f64 / (np - 1) as f64) * 100.0).round() / 100.0).collect();
    let z = pnl_grid(&ctx.strategy, ctx.pricer, &ctx.env, gd, &prices);
    let zjs = format!("[{}]", z.iter().map(|r| js_nums(&r.iter().map(|v| v.round()).collect::<Vec<_>>())).collect::<Vec<_>>().join(","));
    let (xs, ys) = (js_strs(&lab), js_nums(&prices));
    let iv_avg = mean(&ctx.strategy.legs.iter().map(|l| l.iv).collect::<Vec<_>>());
    let step = ((lab.len() as f64 / 12.0).round() as usize).max(1);
    let mut tick_i: Vec<usize> = (0..lab.len()).step_by(step).collect();
    if *tick_i.last().unwrap() != lab.len() - 1 { tick_i.push(lab.len() - 1); }
    let tickvals: Vec<String> = tick_i.iter().map(|&i| lab[i].clone()).collect();

    let mut tr: Vec<String> = Vec::new();
    // 色階涵蓋整個損益範圍：下限 = -成本（最大虧損），上限 = 格點最大獲利；0 固定在中性色
    let zmax = (z.iter().flat_map(|r| r.iter().cloned()).fold(0.0f64, f64::max) / 100.0).ceil() * 100.0;
    let zmin = -ctx.strategy.debit;
    let f0 = (-zmin / (zmax - zmin)).clamp(0.05, 0.95);
    let cs = format!("[[0,'#0a0aa0'],[{:.4},'#6a8cf0'],[{:.4},'#e8e4dc'],[{:.4},'#e0956a'],[{:.4},'#a02020'],[1,'#5a0f0f']]",
                     f0 * 0.5, f0, f0 + (1.0 - f0) * 0.3, f0 + (1.0 - f0) * 0.65);
    tr.push(format!("{{type:'heatmap',x:X,y:Y,z:Z,colorscale:{},zmin:{},zmax:{},name:'P/L',colorbar:{{title:'P/L',thickness:14,len:0.8}}}}",
                    cs, zmin, zmax));
    let contour = |lvl: f64, color: &str, width: f64, dash: &str, name: &str| format!(
        "{{type:'contour',x:X,y:Y,z:Z,showscale:false,showlegend:true,contours:{{start:{lvl},end:{lvl},size:1,coloring:'none'}},line:{{color:'{color}',width:{width},dash:'{dash}'}},name:{}}}",
        js_str(name));
    tr.push(contour(0.0, "black", 2.0, "solid", "解套線 $0"));
    if ctx.exit_spec.breaker && cf.swan_extra_loss > 0.0 {
        tr.push(contour(cf.swan_pnl0 - cf.swan_extra_loss, "#b91c1c", 2.0, "dot", &format!("熔斷線（理論） 總損益 {}", cf.swan_pnl0 - cf.swan_extra_loss)));
    }
    if let Some(tp) = ctx.take_profit_line {
        tr.push(contour(tp, "#1a7f37", 3.0, "dash", &format!("止盈線（自動選定） 總損益 +${}", tp)));
    }
    let last = lab.last().unwrap().clone();
    if let (Some(k), Some(lvl)) = (ctx.exit_spec.atr, atr_lvl) {
        let hwm = ctx.actual.iter().map(|x| x.1).fold(f64::MIN, f64::max);
        tr.push(format!("{{type:'scatter',mode:'lines',x:{},y:[{:.2},{:.2}],line:{{color:'#c026d3',width:3,dash:'dashdot'}},name:{}}}",
            js_strs(&[lab[first].clone(), last.clone()]), lvl, lvl,
            js_str(&format!("ATR停損線 {}×{}（以目前最高收盤 {:.2} 計，創新高會上移）", k, cf.atr_value, hwm))));
    }
    if show_stop { tr.push(format!(
        "{{type:'scatter',mode:'lines',x:{},y:[{},{},{},{},{}],fill:'toself',fillcolor:'rgba(255,215,0,0.3)',line:{{color:'orange',dash:'dash'}},name:{}}}",
        js_strs(&[lab[0].clone(), last.clone(), last.clone(), lab[0].clone(), lab[0].clone()]),
        cf.stop_price, cf.stop_price, cf.swan_price, cf.swan_price, cf.stop_price,
        js_str(&format!("止損警戒區 (${}-${})", cf.swan_price, cf.stop_price)))); }
    if show_stop { tr.push(format!(
        "{{type:'scatter',mode:'lines',x:{xs},y:{},line:{{color:'royalblue',dash:'dash',width:2}},name:{},hoverinfo:'y'}}",
        js_nums(&vec![cf.stop_price; lab.len()]), js_str(&format!("止損線 ${}", cf.stop_price)))); }
    if !env.is_empty() {
        let ex: Vec<String> = env.iter().map(|e| fmt_md(e.0)).collect();
        let ey: Vec<f64> = env.iter().map(|e| (e.1 * 100.0).round() / 100.0).collect();
        let et: Vec<String> = env.iter().map(|e| format!("當日 {} 條熔斷出場：最低 ${:.2} / 最高 ${:.2}", e.3, e.1, e.2)).collect();
        tr.push(format!(
            "{{type:'scatter',mode:'lines+markers',x:{},y:{},text:{},line:{{color:'#7f1d1d',width:3,dash:'dot'}},marker:{{size:6,color:'#7f1d1d'}},name:'熔斷出場每日最低點',hoverinfo:'x+text'}}",
            js_strs(&ex), js_nums(&ey), js_strs(&et)));
    }
    for tl in &ctx.trendlines {
        let v: Vec<f64> = gd.iter().map(|&d| trend_at(tl, d)).collect();
        let mut cnt = 0;
        let keep: Vec<usize> = (0..v.len()).filter(|&i| { if v[i] < y_lo { cnt += 1; } cnt <= 1 }).collect();
        let kx: Vec<String> = keep.iter().map(|&i| lab[i].clone()).collect();
        let ky: Vec<f64> = keep.iter().map(|&i| v[i]).collect();
        tr.push(format!("{{type:'scatter',mode:'lines',x:{},y:{},line:{{color:'#1b7f6b',width:2.5}},name:{}}}",
                        js_strs(&kx), js_nums(&ky), js_str(&tl.name)));
    }
    for rp in &reps {
        let (label, color) = (rp.label.as_str(), rp.color.as_str());
        let r = &results[rp.key];
        let n = r.s.len();
        let a = first.min(n - 1); // 模擬路徑從最新實際收盤那一格接出去，之前的日子只畫實際收盤
        let lx: Vec<String> = r.dates[a..].iter().map(|&d| fmt_md(d)).collect();
        let txt: Vec<String> = r.pnl[a..].iter().map(|p| format!("{}: ${:.0}", label, p)).collect();
        tr.push(format!(
            "{{type:'scatter',mode:'lines+markers',x:{},y:{},name:{},line:{{color:'{color}',width:3}},marker:{{color:'{color}'}},text:{},hoverinfo:'x+y+text'}}",
            js_strs(&lx), js_nums(&r.s[a..]), js_str(&format!("{} n={}", label, rp.cnt)), js_strs(&txt)));
        tr.push(format!(
            "{{type:'scatter',mode:'markers',x:{},y:[{}],marker:{{symbol:'x',size:12,color:'{color}',line:{{color:'white',width:2}}}},text:{},hoverinfo:'text',showlegend:false}}",
            js_strs(&[fmt_md(r.dates[n - 1])]), r.s[n - 1],
            js_strs(&[format!("出場 {} | S=${:.0} | P/L=${:.0}", fmt_md(r.dates[n - 1]), r.s[n - 1], r.pnl[n - 1])])));
    }
    if !ctx.actual.is_empty() {
        let ax: Vec<String> = ctx.actual.iter().map(|&(d, _)| fmt_md(d)).collect();
        let ay: Vec<f64> = ctx.actual.iter().map(|&(_, p)| p).collect();
        let at: Vec<String> = ctx.actual.iter().map(|&(_, p)| format!("{:.2}", p)).collect();
        tr.push(format!(
            "{{type:'scatter',mode:'lines+markers+text',x:{},y:{},text:{},textposition:['top left','top right','bottom left','bottom right'],textfont:{{size:12}},cliponaxis:false,line:{{color:'black',width:3}},marker:{{symbol:'diamond',size:12,color:'white',line:{{color:'black',width:2}}}},name:'實際收盤'}}",
            js_strs(&ax), js_nums(&ay), js_strs(&at)));
    }
    // 標題分兩行，避免窄螢幕（iPad 直向）被截斷；圖例固定在容器底部，底邊距留足
    let title = format!("{}條{} MC模擬（起點為最新實際收盤 ${:.2}）<br>IV {:.0}%｜{}個逐日格點｜情境分類", results.len(), ctx.label, ctx.s0, iv_avg * 100.0, lab.len());
    let layout = format!(
        "{{title:{{text:{},font:{{size:14}},x:0.02,xanchor:'left',y:0.99,yanchor:'top'}},xaxis:{{type:'category',categoryorder:'array',categoryarray:X,tickmode:'array',tickvals:{},ticktext:{},tickangle:-45,automargin:true}},yaxis:{{title:'標的股價 ($)',range:[{},{}],automargin:true}},margin:{{t:80,r:20,b:210,l:60}},legend:{{orientation:'h',x:0,xanchor:'left',yref:'container',y:0.005,yanchor:'bottom',font:{{size:11}}}}}}",
        js_str(&title), js_strs(&tickvals), js_strs(&tickvals), y_lo, y_hi);
    let mut html = String::new();
    write!(html, "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>{}</title>\
<script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script></head><body style=\"margin:0\">\
<div id=\"p\" style=\"width:100%;height:860px;height:max(100dvh,860px)\"></div><script>\
const X={xs},Y={ys},Z={zjs};\nPlotly.newPlot('p',[{}],{},{{responsive:true}});</script></body></html>",
        ctx.label, tr.join(",\n"), layout).unwrap();
    ReportOut { out: Output::Html(html), publish_tp: None }
}

// ---- 文字熱力圖：只用 stdout，Rust Playground 也能看 ------------------------------------------------
// 底色（P/L，以成本計）：@@ 深虧(< -2/3 成本)  %% 中虧(< -1/3 成本)  :: 小虧(< -30)  空白 損益兩平(±30)
//                        .. 小賺(未達止盈)  ++ 達止盈線  ** 賺逾成本
// 疊圖：-- 止損線  == 熔斷價  A..E 各情境代表路徑（A=上漲解套 B=橫盤 C=時間停損 D=跌破停損 E=熔斷）  XX 出場點
fn rep_heatmap_ascii(results: &[Res], ctx: &Ctx) -> ReportOut {
    let cf = &ctx.cfg;
    let gd = &ctx.dates;
    let nd = gd.len();
    let oc = classify_all(results, ctx);
    let first = ctx.actual.len().saturating_sub(1);
    let reps = pick_reps(results, ctx, &oc);
    let env = breaker_env(results, ctx, &oc);
    let mut rep_s: Vec<f64> = reps.iter().flat_map(|rp| results[rp.key].s.clone()).collect();
    rep_s.extend(env.iter().map(|e| e.1));
    let mut y_lo = rep_s.iter().cloned().fold(f64::MAX, f64::min).min(cf.swan_price).min(cf.stop_price).min(ctx.s0);
    let mut y_hi = rep_s.iter().cloned().fold(f64::MIN, f64::max).max(cf.stop_price).max(ctx.s0);
    let pad = 0.05 * (y_hi - y_lo);
    y_lo = (y_lo - pad).floor();
    y_hi = (y_hi + pad).ceil();
    let nr = 36usize;
    let tp = ctx.take_profit_line.unwrap_or(300.0);
    let debit = ctx.strategy.debit;
    let row_price = |r: usize| y_hi - (y_hi - y_lo) * r as f64 / (nr - 1) as f64;
    let row_of = |p: f64| (((y_hi - p) / (y_hi - y_lo)) * (nr - 1) as f64).round().clamp(0.0, (nr - 1) as f64) as usize;
    let mut grid: Vec<Vec<char>> = (0..nr).map(|r| {
        let mut line = Vec::with_capacity(nd * 2);
        for j in 0..nd {
            let v = calc_pnl(&ctx.strategy, row_price(r), gd[j], ctx.pricer, &ctx.env, None);
            let c = if v < -2.0 / 3.0 * debit { '@' } else if v < -debit / 3.0 { '%' } else if v < -30.0 { ':' }
                    else if v <= 30.0 { ' ' } else if v < tp { '.' } else if v < debit { '+' } else { '*' };
            line.push(c);
            line.push(c);
        }
        line
    }).collect();
    let put = |grid: &mut Vec<Vec<char>>, r: usize, j: usize, c: char| { grid[r][2 * j] = c; grid[r][2 * j + 1] = c; };
    let (rs, rw) = (row_of(cf.stop_price), row_of(cf.swan_price));
    let show_stop = !matches!(ctx.exit_spec.stop, StopArg::None);
    if show_stop { for j in 0..nd { put(&mut grid, rw, j, '='); put(&mut grid, rs, j, '-'); } }
    let stopnote = if show_stop { format!("-- 止損線 {}  == 熔斷價 {}；", cf.stop_price, cf.swan_price) } else { String::new() };
    for (ri, rp) in reps.iter().enumerate() {
        let letter = (b'A' + ri as u8) as char;
        let r = &results[rp.key];
        for i in (first + 1)..r.s.len() { put(&mut grid, row_of(r.s[i]), i, letter); }
        let n = r.s.len();
        put(&mut grid, row_of(r.s[n - 1]), n - 1, 'X');
    }
    for e in &env { if let Some(j) = gd.iter().position(|&d| d == e.0) { put(&mut grid, row_of(e.1), j, 'v'); } }
    for (i, &(_, p)) in ctx.actual.iter().enumerate() { put(&mut grid, row_of(p), i, 'O'); }
    let mut out = String::new();
    writeln!(out, "{}條{} MC模擬 | 縱軸=標的股價 橫軸=日期（逐日格點）| 止盈線 +${}", results.len(), ctx.label, tp).unwrap();
    for r in 0..nr {
        let line: String = grid[r].iter().collect();
        writeln!(out, "{:>7.1} |{}", row_price(r), line).unwrap();
    }
    let mut axis = vec![' '; nd * 2 + 2];
    for j in (0..nd).step_by(3) {
        for (o, ch) in fmt_md(gd[j]).chars().enumerate() {
            if 2 * j + o < axis.len() { axis[2 * j + o] = ch; }
        }
    }
    writeln!(out, "        +{}", "-".repeat(nd * 2)).unwrap();
    writeln!(out, "         {}", axis.iter().collect::<String>().trim_end()).unwrap();
    writeln!(out, "圖例：底色 @@深虧 %%中虧 ::小虧 空白≈兩平 ..小賺 ++達止盈線(+${}) **賺逾成本(+${})；{}XX 出場點  OO 實際收盤  vv 熔斷出場每日最低點", tp, debit, stopnote).unwrap();
    for (ri, rp) in reps.iter().enumerate() {
        let r = &results[rp.key];
        let n = r.s.len();
        writeln!(out, "  {}  {} n={}：{} 出場 S=${:.1} P/L=${:.0}", (b'A' + ri as u8) as char, rp.label, rp.cnt,
                 fmt_md(r.dates[n - 1]), r.s[n - 1], r.pnl[n - 1]).unwrap();
    }
    ReportOut { out: Output::Text(out), publish_tp: None }
}

// =============================================================================
// PART 3：交易定義（新部位只需要改這一段）
// =============================================================================
// true = 在輸出最後印出完整 HTML 原始碼（Rust Playground 用：複製存成 .html 即可開啟互動熱力圖）
const PRINT_HTML: bool = true;
const SIM_VOL: f64 = 1.19; // 路徑年化波動
const SIM_DRIFT: f64 = 0.05; // 路徑年化 drift

// 券商矩陣：2026-10-20 截圖（SOXL 98.76，IV 設定「平均」110%，+57.50 未實現損益）。
// 實測各履約價 IV 近乎平坦 → 採固定履約價（skew = 0）；舊假設 skew = 1.2 只列為敏感度情境。
const MARKET: &str = "
price 10/20 10/26 10/30 11/4 11/9 11/13 11/19 11/23 11/27 12/3 12/8 12/14 12/19
145 629 663 688 723 761 796 855 902 956 1058 1174 1395 1744
139 573 605 628 660 696 729 785 829 881 979 1091 1310 1812
134 521 551 572 602 636 666 718 759 806 896 998 1194 1474
128 454 480 499 526 555 581 626 661 702 777 860 1006 1100
123 394 417 433 456 481 504 542 571 604 663 725 817 815
117 317 336 349 367 386 403 431 452 474 512 546 571 508
112 249 264 274 287 302 314 333 346 359 378 387 366 283
106 164 174 180 188 195 201 209 213 216 212 197 137 52
100 76 80 83 85 86 86 83 79 71 50 15 -62 -139
95 2 1 1 -2 -5 -10 -20 -31 -46 -78 -122 -200 -267
89 -88 -93 -97 -104 -113 -122 -140 -156 -176 -217 -263 -333 -386
84 -162 -170 -177 -187 -199 -211 -233 -252 -274 -316 -359 -418 -460
78 -247 -258 -267 -280 -294 -308 -333 -352 -374 -413 -449 -492 -521
73 -314 -327 -336 -350 -366 -380 -404 -422 -442 -475 -503 -535 -555
67 -387 -400 -410 -424 -439 -452 -473 -488 -504 -528 -548 -568 -579
61 -450 -463 -472 -484 -497 -508 -525 -536 -547 -563 -575 -586 -592
56 -494 -505 -513 -523 -534 -542 -555 -563 -571 -581 -588 -594 -597
";

// =============================================================================
// 每日更新的資料（只改這一段，或在同資料夾放 config.txt 覆蓋；格式相同）
// 每行 key = value；# 開頭為註解。close 可重複，每天新增一行即可。
// =============================================================================
const DEFAULT_CONFIG: &str = "
label = SOXL
# --- 建倉：日期、成本、各腿（數量 履約價C/P 到期日；leg 與 iv 的順序一致）---
entry_date = 2026-10-20
debit = 600
leg = +1 125C 2027-01-15
leg = -1 140C 2026-12-18
# --- 實際收盤（日期 價格），建倉日起每個交易日新增一行 ---
close = 2026-10-20 97.55
close = 2026-10-21 103.24
close = 2026-10-22 82.11
close = 2026-10-23 83.56
close = 2026-10-26 90.50
# --- 隱含波動率：每次更新週期新增一行 iv = 生效日 各腿IV（1.16 = 116%）---
# 模擬用「最新收盤日當時生效」的那組；早於第一行的日期沿用第一行
iv = 2026-10-22 1.203 1.207
iv = 2026-10-23 1.16 1.22
# --- 模擬設定 ---
n_paths = 10000
drift = 0.05
# vol 不寫 = 取各腿 IV 平均
# --- 出場規則 ---
# ATR 停損（吊燈式：建倉以來最高收盤 - k×ATR，只在收盤判斷）。atr_k = 0 代表不啟用；atr_scan 只用在停損規則掃描比較
atr_value = 15
atr_k = 2.5
atr_scan = 1 1.5 2 2.5 3
window_tp = 100
time_stop_date = 2026-11-06
swan_price = 78
swan_extra_loss = 250
short_term_date = 2026-10-30
match_days = 5
";
// 驗證一條 leg：「數量 履約價C|P 到期日」，例如 +1 125C 2027-01-15（避免後面 strategy_from_text 解析失敗而 panic）
fn validate_leg(v: &str) -> Result<(), String> {
    let f: Vec<&str> = v.split_whitespace().collect();
    if f.len() != 3 { return Err(format!("leg 格式應為「數量 履約價C 到期日」: {}", v)); }
    match f[0].parse::<f64>() { Ok(q) if q.is_finite() && q != 0.0 => {} _ => return Err(format!("leg 數量不正確: {}", f[0])) }
    let kind = f[1].as_bytes()[f[1].len() - 1];
    if kind != b'C' && kind != b'P' { return Err(format!("leg 履約價需以 C 或 P 結尾: {}", f[1])); }
    match f[1][..f[1].len() - 1].parse::<f64>() { Ok(k) if k.is_finite() && k > 0.0 => {} _ => return Err(format!("leg 履約價不正確: {}", f[1])) }
    try_date(f[2]).map(|_| ())
}
fn apply_config(ctx: &mut Ctx, text: &str, max_closes: Option<usize>) -> Result<(), String> {
    let mut closes: Vec<(Day, f64)> = Vec::new();
    let (mut ivl, mut ivs) = (None, None);
    let mut ivh: Vec<(Day, Vec<f64>)> = Vec::new();
    let mut legs: Vec<String> = Vec::new();
    let mut entry: Option<Day> = None;
    let mut atr_k: Option<f64> = None;
    let mut atr_scan: Vec<f64> = Vec::new();
    let mut vol: Option<f64> = None;
    let mut cap: Option<f64> = None;
    for (ln, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap().trim();
        if line.is_empty() { continue; }
        let (k, v) = line.split_once('=').ok_or(format!("第 {} 行缺少 '=': {}", ln + 1, raw))?;
        let (k, v) = (k.trim(), v.trim());
        let num = |x: &str| x.parse::<f64>().ok().filter(|v| v.is_finite()).ok_or_else(|| format!("第 {} 行數字格式錯誤: {}", ln + 1, raw));
        let dt = |x: &str| try_date(x).map_err(|e| format!("第 {} 行：{}", ln + 1, e));
        let pos = |x: &str| num(x).and_then(|v| if v > 0.0 { Ok(v) } else { Err(format!("第 {} 行必須大於 0: {}", ln + 1, raw)) });
        let nonneg = |x: &str| num(x).and_then(|v| if v >= 0.0 { Ok(v) } else { Err(format!("第 {} 行不可為負: {}", ln + 1, raw)) });
        match k {
            "close" => {
                let mut it = v.split_whitespace();
                let d = it.next().ok_or(format!("第 {} 行缺少日期", ln + 1))?;
                let p = it.next().ok_or(format!("第 {} 行缺少價格", ln + 1))?;
                closes.push((dt(d)?, pos(p)?));
            }
            "label" => ctx.label = v.to_string(),
            "atr_value" => ctx.cfg.atr_value = nonneg(v)?,
            "atr_k" => atr_k = Some(nonneg(v)?),
            "atr_scan" => { atr_scan = v.split_whitespace().map(|x| pos(x)).collect::<Result<Vec<_>, _>>()?; }
            "debit" => ctx.debit = pos(v)?,
            "entry_date" => entry = Some(dt(v)?),
            "leg" => { validate_leg(v).map_err(|e| format!("第 {} 行：{}", ln + 1, e))?; legs.push(v.to_string()) }
            "iv" => {
                let f: Vec<&str> = v.split_whitespace().collect();
                if f.len() < 2 { return Err(format!("第 {} 行 iv 格式：iv = 日期 各腿IV（依 leg 順序）", ln + 1)); }
                let vals: Result<Vec<f64>, String> = f[1..].iter().map(|x| pos(x)).collect();
                ivh.push((dt(f[0])?, vals?));
            }
            "iv_long" => ivl = Some(pos(v)?),
            "iv_short" => ivs = Some(pos(v)?),
            "n_paths" => { let n = pos(v)?; if n > 5.0e7 { return Err(format!("第 {} 行 n_paths 過大（上限 5000 萬）", ln + 1)); } ctx.n_paths = n as usize; ctx.oos_n = ctx.n_paths; }
            "drift" => ctx.drift = num(v)?,
            "vol" => vol = Some(pos(v)?),
            "window_tp" => cap = Some(nonneg(v)?),
            "time_stop_date" => ctx.cfg.time_stop_date = dt(v)?,
            "swan_price" => ctx.cfg.swan_price = num(v)?,
            "swan_extra_loss" => ctx.cfg.swan_extra_loss = nonneg(v)?,
            "stop_price" => ctx.cfg.stop_price = num(v)?,
            "short_term_date" => ctx.short_term_date = dt(v)?,
            "match_days" => { let n = pos(v)?; if n > 60.0 { return Err(format!("第 {} 行 match_days 上限 60", ln + 1)); } ctx.match_days = n as usize }
            other => return Err(format!("未知設定項: {}", other)),
        }
    }
    if !closes.is_empty() {
        closes.sort_by_key(|&(d, _)| d);
        for w in closes.windows(2) { if w[0].0 == w[1].0 { return Err(format!("close 日期重複：{}", fmt_md(w[0].0))); } }
        if let Some(&(d, _)) = closes.iter().find(|&&(d, _)| iso_weekday(d) > 5) { return Err(format!("close 日期 {} 不是交易日（週末）", fmt_md(d))); }
        if let Some(k) = max_closes { closes.truncate(k.max(1)); }
        if let Some(ed) = entry {
            if closes[0].0 != ed { return Err(format!("entry_date {} 與第一筆 close {} 不一致", fmt_md(ed), fmt_md(closes[0].0))); }
        }
        ctx.today = closes[0].0;
        ctx.sim_start = ctx.today + 1;
        ctx.spot = closes[0].1;
        ctx.env.s0_ref = closes[closes.len() - 1].1;
        ctx.actual = closes;
    } else {
        if let (Some(ed), Some(&(d0, _))) = (entry, ctx.actual.first()) {
            if d0 != ed { return Err(format!("entry_date {} 與第一筆 close {} 不一致", fmt_md(ed), fmt_md(d0))); }
        }
    }
    if !legs.is_empty() { ctx.spec = legs.join("; "); }
    if !ivh.is_empty() { ivh.sort_by_key(|x| x.0); ctx.iv_hist = ivh; }
    else if let (Some(a), Some(b)) = (ivl, ivs) { ctx.iv_hist = vec![(Day::MIN, vec![a, b])]; }
    else if ctx.iv_hist.is_empty() { ctx.iv_hist = vec![(Day::MIN, ctx.iv_init.clone())]; }
    let asof = ctx.actual.last().map(|x| x.0).unwrap_or(ctx.today);
    ctx.iv_init = iv_at(&ctx.iv_hist, asof);
    let nleg = ctx.spec.split(';').count();
    if ctx.iv_init.len() != nleg { return Err(format!("IV 數量（{}）與 leg 數量（{}）不一致", ctx.iv_init.len(), nleg)); }
    let avg = ctx.iv_init.iter().sum::<f64>() / ctx.iv_init.len() as f64;
    ctx.vol = vol.unwrap_or(avg);
    ctx.env.sigma = ctx.vol;
    if let Some(c) = cap {
        let md = fmt_md(ctx.cfg.time_stop_date);
        let fix = |r: &mut RuleSpec| {
            if r.window_tp.is_some() { r.window_tp = Some(c); }
            r.name = r.name.replace("0~+100", &format!("0~+{}", c)).replace("11/06", &md);
        };
        fix(&mut ctx.exit_spec);
        for r in ctx.stop_variants.iter_mut().chain(ctx.sens_rules.iter_mut()).chain(ctx.tp_variants.iter_mut()).chain(ctx.drift_rules.iter_mut()) { fix(r); }
    }
    if let Some(k) = atr_k {
        ctx.exit_spec.atr = if k > 0.0 { Some(k) } else { None };
        ctx.cfg.atr_k = ctx.exit_spec.atr;
    }
    let base_atr = ctx.exit_spec.atr;
    for r in ctx.sens_rules.iter_mut().chain(ctx.drift_rules.iter_mut()).chain(ctx.tp_variants.iter_mut()) {
        if r.name.starts_with("現行") { r.atr = base_atr; }
    }
    if let Some(r0) = ctx.stop_variants.first_mut() { if r0.name.starts_with("現行") { r0.atr = base_atr; } }
    if !atr_scan.is_empty() {
        ctx.stop_variants.retain(|r| !r.name.starts_with("現行+ATR") && !r.name.starts_with("現行去掉ATR"));
        if base_atr.is_some() {
            let mut r = ctx.exit_spec.clone(); r.atr = None; r.name = "現行去掉ATR(對照)".into();
            ctx.stop_variants.push(r);
        }
        for k in atr_scan {
            let mut r = ctx.exit_spec.clone(); r.atr = Some(k); r.name = format!("現行+ATR停損 k={}", k);
            ctx.stop_variants.push(r);
        }
    }
    Ok(())
}

fn trade() -> Ctx {
    let dummy = Strategy { debit: 0.0, legs: vec![] };
    Ctx {
        label: "SOXL".into(),
        spec: "+1 125C 2027-01-15; -1 140C 2026-12-18".into(),
        debit: 600.0, spot: 97.55, iv_init: vec![1.16, 1.22], iv_hist: vec![],
        today: date("2026-10-20"), sim_start: date("2026-10-21"), // today = 券商矩陣第一欄（估值日）
        short_term_date: date("2026-10-28"),
        pricer: PricerKind::Bs,
        env: Env { r: 0.045, sigma: 1.19, skew: 0.0, s0_ref: 83.56, vf: 1.0, term_a: 0.0, term_tau: 0.1 },
        path_model: PathKind::Normal, vol: SIM_VOL, drift: SIM_DRIFT, n_paths: 10000,
        sv: Sv { kappa: 3.0, xi: 1.2, rho: -0.6 },
        seed: SeedFn { base: 1000, mult: 79 },
        oos_seed: SeedFn { base: 2000, mult: 83 }, oos_n: 10000,
        cfg: Cfg {
            stop_price: 85.0, swan_price: 78.0, swan_extra_loss: 250.0, bull_price: 107.0,
            swan_loss_from_today: false, // 熔斷 = 總虧損（相對成本）超過 250；true 則從今日損益起算
            time_stop_date: date("2026-11-06"),
            jump_lambda: 0.73, jump_size: -0.15, open_shock_sd: 0.025, pnl_big: 300.0, atr_k: None, atr_value: 15.0,
            horizon: 0, pnl_today: 0.0, swan_pnl0: 0.0,
        },
        outcome: OutcomeKind::HoldNoStop,
        exit_spec: RuleSpec::new("現行規則").no_stop().window_tp(100.0), // 價格停損關閉；11/06 起損益 0~+100 之間止盈，<0 時間停損；虧損逾 250 熔斷
        trendlines: vec![],
        market_year: 2026,
        market: String::new(), // 沒有新的券商矩陣：不對帳、不校準，直接採用實測 IV
        // 停損規則比較（第一個為基準）
        stop_variants: vec![
            RuleSpec::new("現行:11/06起區間止盈(0~+100)+時間停損+熔斷").no_stop().window_tp(100.0),
            RuleSpec::new("舊規則:無區間止盈").no_stop(),
            RuleSpec::new("停損75").stop(75.0).window_tp(100.0),
            RuleSpec::new("停損70").stop(70.0),
            RuleSpec::new("無時間停損(留熔斷)").no_stop().no_time_stop(),
            RuleSpec::new("完全不設停損(持有到短腿到期)").no_stop().no_time_stop().no_breaker(),
        ],
        sens_rules: vec![
            RuleSpec::new("現行規則").no_stop().window_tp(100.0),
            RuleSpec::new("不設停損").no_stop().no_time_stop().no_breaker(),
        ],
        auto_tp: Some(AutoTp {
            grid: (0..9).map(|k| 100.0 + 50.0 * k as f64).collect(),
            objective: Objective::LossAverse, lambda: 1.5, slippage: 20.0, se_k: 1.0, prefer: Prefer::Higher,
        }),
        tp_variants: vec![
            RuleSpec::new("現行(11/06起區間止盈)").no_stop().window_tp(100.0),
            RuleSpec::tp("止盈+100", 100.0).no_stop(), RuleSpec::tp("止盈+150", 150.0).no_stop(),
            RuleSpec::tp("止盈+200", 200.0).no_stop(), RuleSpec::tp("止盈+300", 300.0).no_stop(),
        ],
        drift_rules: vec![
            RuleSpec::new("現行規則").no_stop().window_tp(100.0),
            RuleSpec::new("無時間停損").no_stop().no_time_stop(),
            RuleSpec::new("不設停損").no_stop().no_time_stop().no_breaker(),
            RuleSpec::tp("止盈+100", 100.0).no_stop(), RuleSpec::tp("止盈+150", 150.0).no_stop(),
            RuleSpec::tp("止盈+200", 200.0).no_stop(), RuleSpec::tp("止盈+300", 300.0).no_stop(),
        ],
        sens_n: 500,
        sens_scenarios: {
            let sc = Scenario::new;
            let mut v = vec![
                sc("基準:市場微笑(固定履約價)"),
                sc("舊假設:偏斜1.2"),
                sc("低波動90%"),
                sc("高波動140%"),
                sc("偏斜1.2+隨機波動+期限結構"),
                sc("市場微笑:固定履約價+隨機波動"),
            ];
            v[1].skew = Some(1.2);
            v[2].vol = Some(0.90); v[2].iv_scale = 0.90 / SIM_VOL;
            v[3].vol = Some(1.40); v[3].iv_scale = 1.40 / SIM_VOL;
            v[4].pricer = PricerKind::BsDyn; v[4].path = PathKind::Sv; v[4].skew = Some(1.2); v[4].term_a = 0.5;
            v[5].pricer = PricerKind::BsDyn; v[5].path = PathKind::Sv; v[5].skew = Some(0.0);
            v
        },
        drift_grid: vec![-1.0, -0.5, 0.0, 0.5, 1.0], drift_n: 500,
        reports: vec!["three_day_up_probability", "outcome_probabilities", "expected_value",
                      "stop_rule_scan", "stop_rule_scan_oos",
                      "auto_take_profit", "sensitivity_scenarios", "breaker_envelope", "heatmap_ascii", "heatmap_typical_representative"],
        heavy_reports: vec!["drift_scan"],
        take_profit_line: None,
        daily_grid: true,
        actual: vec![(date("2026-10-20"), 97.55), (date("2026-10-21"), 103.24), (date("2026-10-22"), 82.11), (date("2026-10-23"), 83.56)],
        now: 0,
        match_days: 5,
        strategy: dummy, s0: 0.0, dates: vec![],
    }
}

// =============================================================================
// PART 4：執行與輸出
// =============================================================================
fn run_replay(n_closes: usize, file_txt: &str, out_path: &str, _heavy: bool) {
    QUIET.store(true, Ordering::Relaxed);
    let mut table_rows: Vec<Vec<String>> = Vec::new();
    let mut headers: Vec<String> = Vec::new();
    for k in 1..=n_closes {
        let mut c = trade();
        apply_config(&mut c, DEFAULT_CONFIG, Some(k)).unwrap();
        if !file_txt.is_empty() { apply_config(&mut c, file_txt, Some(k)).unwrap(); }
        setup_trade(&mut c);
        let (d, p) = *c.actual.last().unwrap();
        let st = strategy_at(&c, d);
        let pnl = calc_pnl(&st, p, d, c.pricer, &c.env, None);
        let status = rep_actual_path(&c, false).map_or("持有".to_string(), |(dd, r)| format!("{}已觸發{}", fmt_md(dd), r));
        let (_r, outs) = run(&mut c, &["outcome_probabilities", "expected_value"]);
        let mut probs: Vec<String> = Vec::new();
        let (mut ev, mut win) = ("NA".to_string(), "NA".to_string());
        for (name, _t, o) in &outs {
            if let Output::Table(t) = &o.out {
                if *name == "outcome_probabilities" { probs = t.rows.iter().map(|r| r[1].clone()).collect(); }
                else {
                    for r in &t.rows {
                        if r[0].starts_with("期望值") { ev = r[1].clone(); }
                        if r[0].starts_with("勝率") { win = r[1].clone(); }
                    }
                }
            }
        }
        if headers.is_empty() {
            headers = vec!["日期".into(), "收盤".into(), "實際損益".into()];
            headers.extend(c.outcome.levels().iter().map(|l| format!("{}%", l)));
            headers.extend(["期望值".to_string(), "勝率%".to_string(), "實際路徑狀態".to_string()]);
        }
        let mut row = vec![fmt_md(d), f2(p), f1(pnl)];
        row.extend(probs);
        row.extend([ev, win, status]);
        table_rows.push(row);
    }
    QUIET.store(false, Ordering::Relaxed);
    let csv: String = std::iter::once(headers.join(",")).chain(table_rows.iter().map(|r| r.join(","))).collect::<Vec<_>>().join("\n");
    match std::fs::write(out_path, csv + "\n") {
        Ok(_) => println!("逐日回放結果已輸出：{}", out_path),
        Err(e) => eprintln!("無法寫入 {}：{}", out_path, e),
    }
    println!("\n--- 逐日回放（每天只用到當天為止的實際收盤重新模擬）---");
    Table { headers, rows: table_rows }.print();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut ctx = trade();
    let mut run_heavy = true;
    let mut out_path = "heatmap.html".to_string();
    let mut cfg_path = "config.txt".to_string();
    let mut replay = false;
    let mut replay_out = "replay.csv".to_string();
    let mut cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--no-heavy" => run_heavy = false,
            "--sparse" => ctx.daily_grid = false,
            "--out" => { i += 1; out_path = args[i].clone(); }
            "--config" => { i += 1; cfg_path = args[i].clone(); }
            "--replay" => replay = true,
            "--replay-out" => { i += 1; replay_out = args[i].clone(); }
            "--print-config" => { println!("{}", DEFAULT_CONFIG.trim()); return; }
            "--cores" => { i += 1; cores = args[i].parse().unwrap(); }
            other => eprintln!("忽略未知參數: {}", other),
        }
        i += 1;
    }
    NCORES.store(cores, Ordering::Relaxed);

    if let Err(e) = apply_config(&mut ctx, DEFAULT_CONFIG, None) { eprintln!("內建設定錯誤: {}", e); std::process::exit(1); }
    let file_txt = std::fs::read_to_string(&cfg_path).ok();
    match &file_txt {
        Some(t) => {
            if let Err(e) = apply_config(&mut ctx, t, None) { eprintln!("{} 設定錯誤: {}", cfg_path, e); std::process::exit(1); }
            println!("已讀取設定檔：{}（最新實際收盤 {}）", cfg_path, ctx.actual.last().map(|&(d, p)| format!("{} ${:.2}", fmt_md(d), p)).unwrap_or_default());
        }
        None => println!("未找到 {}，使用內建設定（DEFAULT_CONFIG）", cfg_path),
    }

    if replay {
        run_replay(ctx.actual.len(), file_txt.as_deref().unwrap_or(""), &replay_out, run_heavy);
        return;
    }

    setup_trade(&mut ctx);
    println!("\n--- 建倉日以來的實際路徑 ---");
    rep_actual_path(&ctx, true);
    let t0 = Instant::now();
    let mut names: Vec<&str> = ctx.reports.clone();
    if run_heavy { names.extend(ctx.heavy_reports.iter()); }
    let (_results, outs) = run(&mut ctx, &names);

    let mut html: Option<String> = None;
    for (name, title, o) in outs {
        match o.out {
            Output::Table(t) => {
                println!("\n--- {} ---", if title.is_empty() { name } else { title });
                t.print();
            }
            Output::Html(h) => { if html.is_none() { html = Some(h); } }
            Output::Text(t) => { println!("\n--- {} ---", title); println!("{}", t); }
        }
    }
    println!("\n總執行時間：{:.2} 分鐘（格點數 {}，核心數 {}）",
             t0.elapsed().as_secs_f64() / 60.0, ctx.dates.len(), cores);
    if let Some(h) = html {
        if PRINT_HTML {
            println!("\n===== 複製下面 BEGIN 與 END 之間的內容，存成 heatmap.html 後用瀏覽器開啟 =====");
            println!("----- BEGIN HTML -----");
            println!("{}", h);
            println!("----- END HTML -----");
        }
        match std::fs::write(&out_path, h) {
            Ok(_) => println!("互動式熱力圖已輸出：{}（本機用瀏覽器開啟；Playground 請看上方文字熱力圖）", out_path),
            Err(e) => println!("無法寫入 {}：{}（可忽略，文字熱力圖已在上方）", out_path, e),
        }
    }
}

// =============================================================================
// PART 5：驗證
//   - props：同一份「性質」同時給 cargo test（窮舉網格）與 Kani（符號輸入、對所有輸入證明）使用
//   - tests：cargo test
//   - verification：cargo kani（需要先安裝 Kani；見 Colab 筆記本）
// 這些純函數沒有 Ctx、字串、集合，所以 Kani 能處理；Black-Scholes 的 exp/ln/sqrt 不在驗證範圍內。
// =============================================================================
#[cfg(any(test, kani))]
mod props {
    use super::*;

    // 日期：z 是合法天數
    pub fn civil_roundtrip(z: Day) -> bool {
        let (y, m, d) = civil_from_days(z);
        (1..=12).contains(&m) && d >= 1 && d <= days_in_month(y, m) && days_from_civil(y, m, d) == z
    }
    pub fn ymd_roundtrip(y: i32, m: u32, d: u32) -> bool {
        civil_from_days(days_from_civil(y, m, d)) == (y, m, d)
    }
    pub fn weekday_ok(z: Day) -> bool {
        let w = iso_weekday(z);
        (1..=7).contains(&w) && iso_weekday(z + 1) == w % 7 + 1
    }
    // parse_ymd 成功就一定是存在的日期，且能還原回同一天
    pub fn parse_ymd_ok(b: &[u8]) -> bool {
        match parse_ymd(b) {
            None => true,
            Some((y, m, d)) => (0..=9999).contains(&y) && (1..=12).contains(&m) && d >= 1 && d <= days_in_month(y, m)
                && civil_from_days(days_from_civil(y, m, d)) == (y, m, d),
        }
    }

    // 出場規則：理由與條件一致、沒有遺漏、到期必出場（輸入皆為非 NaN）
    pub fn ladder_consistent(p: &LadderParams, d: Day, s: f64, pnl: f64, hwm: f64) -> bool {
        let c_stop = p.stop.map_or(false, |x| s < x);
        let c_br = p.br && ((pnl - p.pnl0) < -p.extra || s < p.swan_px);
        let c_atr = p.atr_d.map_or(false, |a| s < hwm - a);
        let c_ts = p.ts && d >= p.ts_date && pnl < 0.0;
        let c_wtp = p.wtp.map_or(false, |cap| d >= p.ts_date && pnl >= 0.0 && pnl <= cap);
        let c_exp = d >= p.horizon;
        let r = ladder_reason(p, d, s, pnl, hwm);
        let any = c_stop || c_br || c_atr || c_ts || c_wtp || c_exp;
        if r.is_some() != any { return false; }
        if c_exp && r.is_none() { return false; }
        match r {
            None => true,
            Some(ExitReason::PriceStop) => c_stop,
            Some(ExitReason::Breaker) => !c_stop && c_br,
            Some(ExitReason::Atr) => !c_stop && !c_br && c_atr,
            Some(ExitReason::TimeStop) => !c_stop && !c_br && !c_atr && c_ts,
            Some(ExitReason::WindowTp) => !c_stop && !c_br && !c_atr && !c_ts && c_wtp,
            Some(ExitReason::Expiry) => !c_stop && !c_br && !c_atr && !c_ts && !c_wtp && c_exp,
        }
    }
    // 只開「時間停損 + 區間止盈」、其餘規則關閉、已過時間停損日但未到期：只有損益 > cap 才續抱
    pub fn window_semantics(ts_date: Day, horizon: Day, cap: f64, d: Day, pnl: f64) -> bool {
        let p = LadderParams { horizon, stop: None, ts: true, ts_date, br: false, wtp: Some(cap), extra: 0.0, pnl0: 0.0, swan_px: 0.0, atr_d: None };
        let r = ladder_reason(&p, d, 100.0, pnl, 100.0);
        r.is_some() == (pnl <= cap)
    }
    // 時間停損日之前、未到期、無價格/熔斷/ATR 規則：時間停損與區間止盈都不會觸發
    pub fn before_time_stop_quiet(ts_date: Day, horizon: Day, cap: f64, d: Day, pnl: f64) -> bool {
        let p = LadderParams { horizon, stop: None, ts: true, ts_date, br: false, wtp: Some(cap), extra: 0.0, pnl0: 0.0, swan_px: 0.0, atr_d: None };
        ladder_reason(&p, d, 100.0, pnl, 100.0).is_none()
    }
    // 創新高（收盤 = 最高收盤）時，只要 ATR 距離 >= 0，ATR 停損不會觸發
    pub fn atr_not_at_new_high(a: f64, s: f64) -> bool {
        let p = LadderParams { horizon: Day::MAX, stop: None, ts: false, ts_date: Day::MAX, br: false, wtp: None, extra: 0.0, pnl0: 0.0, swan_px: 0.0, atr_d: Some(a) };
        ladder_reason(&p, 0, s, 0.0, s).is_none()
    }

    // 情境分類：每筆輸入恰好一類、類別範圍、各類別的必要條件
    pub fn classify_ok(p: &HoldParams, s_end: f64, pnl_end: f64, d_end: Day, hwm: f64) -> bool {
        let c = classify_hold(p, s_end, pnl_end, d_end, hwm);
        let breaker = pnl_end < p.swan_pnl0 - p.swan_extra_loss;
        let early = d_end < p.horizon;
        c <= 6
            && ((c == 4) == breaker)
            && (breaker || (early == (c == 3 || c == 5 || c == 6)))
            && (c != 6 || (p.atr_k.is_some() && s_end < hwm - p.atr_k.unwrap() * p.atr_value))
            && (c != 5 || (pnl_end >= 0.0 && !p.atr_k.map_or(false, |k| s_end < hwm - k * p.atr_value)))
            && (c != 3 || pnl_end < 0.0)
            && (c != 0 || (!early && pnl_end >= p.pnl_big))
            && (c != 1 || (!early && pnl_end >= 0.0 && pnl_end < p.pnl_big))
            && (c != 2 || (!early && pnl_end < 0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- 日期 ----
    #[test]
    fn date_roundtrip_exhaustive() {
        for z in -25567..110000 { assert!(props::civil_roundtrip(z), "z={}", z); assert!(props::weekday_ok(z), "z={}", z); }
        for y in 1900..=2300 { for m in 1..=12 { for d in 1..=days_in_month(y, m) { assert!(props::ymd_roundtrip(y, m, d), "{}-{}-{}", y, m, d); } } }
    }
    #[test]
    fn date_known_values() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(iso_weekday(0), 4); // 1970-01-01 是週四
        assert_eq!(iso_weekday(date("2026-10-20")), 2); // 週二
        assert_eq!(iso_weekday(date("2026-10-24")), 6); // 週六
        assert_eq!(date("2027-01-15") - date("2026-12-18"), 28);
        assert_eq!(fmt_md(date("2026-11-06")), "11/06");
        assert_eq!(days_in_month(2024, 2), 29);
        assert_eq!(days_in_month(2100, 2), 28);
        assert_eq!(days_in_month(2000, 2), 29);
    }
    #[test]
    fn parse_ymd_table() {
        let bad = ["", "2026-1-5", "2026/10/20", "2026-10-2", "2026-13-01", "2026-00-10", "2026-02-30", "2025-02-29",
                   "2026-10-00", "2026-10-32", "abcd-ef-gh", " 2026-10-20", "2026-10-20 ", "2026-10-2０", "２０２６-10-20", "2026-10-20\n"];
        for b in bad { assert!(parse_ymd(b.as_bytes()).is_none(), "應拒絕: {:?}", b); assert!(try_date(b).is_err()); }
        assert_eq!(parse_ymd(b"2024-02-29"), Some((2024, 2, 29)));
        assert_eq!(parse_ymd(b"2026-12-31"), Some((2026, 12, 31)));
    }
    #[test]
    fn parse_ymd_never_panics_on_any_10_bytes_sample() {
        // 對隨機 10 位元組（LCG）與「合法格式 + 單一位元組變異」都不 panic，且成功時性質成立
        let mut x: u64 = 88172645463325252;
        for _ in 0..200_000 {
            let mut b = [0u8; 10];
            for v in b.iter_mut() { x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *v = (x >> 56) as u8; }
            assert!(props::parse_ymd_ok(&b));
        }
        let base = *b"2026-10-20";
        for i in 0..10 { for c in 0..=255u8 { let mut b = base; b[i] = c; assert!(props::parse_ymd_ok(&b)); } }
        for n in 0..14 { let v = vec![b'0'; n]; assert!(props::parse_ymd_ok(&v)); }
    }

    // ---- 出場規則 ----
    #[test]
    fn ladder_consistency_grid() {
        let (stops, wtps, atrs) = ([None, Some(80.0)], [None, Some(0.0), Some(100.0)], [None, Some(30.0)]);
        let (ds, ss, pnls) = ([-1, 0, 39, 40, 41, 99, 100, 101], [50.0, 77.9, 78.0, 85.0, 100.0, 200.0],
                              [-500.0, -251.0, -250.0, -249.99, -1.0, -0.0001, 0.0, 0.0001, 50.0, 100.0, 100.0001, 500.0]);
        let mut n = 0u64;
        for &stop in &stops { for ts in [false, true] { for br in [false, true] { for &wtp in &wtps { for &atr_d in &atrs {
            for pnl0 in [0.0, -100.0] { for swan_px in [0.0, 78.0] {
                let p = LadderParams { horizon: 100, stop, ts, ts_date: 40, br, wtp, extra: 250.0, pnl0, swan_px, atr_d };
                for &d in &ds { for &s in &ss { for &pnl in &pnls { for hwm in [s, s + 10.0, s + 30.0, s + 31.0] {
                    assert!(props::ladder_consistent(&p, d, s, pnl, hwm), "d={} s={} pnl={} hwm={}", d, s, pnl, hwm);
                    n += 1;
                }}}}
            }}
        }}}}}
        assert!(n > 100_000);
    }
    #[test]
    fn window_semantics_grid() {
        for cap in [0.0, 1.0, 100.0, 1000.0] {
            for d in [40, 41, 60, 99] { for pnl in [-1000.0, -1.0, -0.0001, 0.0, 0.0001, 99.9999, 100.0, 100.0001, 1000.0] {
                assert!(props::window_semantics(40, 100, cap, d, pnl), "cap={} d={} pnl={}", cap, d, pnl);
            }}
            for d in [-5, 0, 39] { for pnl in [-1000.0, -1.0, 0.0, 50.0, 1000.0] {
                assert!(props::before_time_stop_quiet(40, 100, cap, d, pnl));
            }}
        }
        for a in [0.0, 0.5, 15.0, 100.0] { for s in [1.0, 83.56, 500.0] { assert!(props::atr_not_at_new_high(a, s)); } }
    }
    #[test]
    fn atr_chandelier_numbers() {
        // 最高收盤 103.24、ATR 15、k=2.5：停損 = 103.24 - 37.5 = 65.74
        let p = LadderParams { horizon: Day::MAX, stop: None, ts: false, ts_date: Day::MAX, br: false, wtp: None, extra: 0.0, pnl0: 0.0, swan_px: 0.0, atr_d: Some(2.5 * 15.0) };
        assert_eq!(ladder_reason(&p, 0, 65.75, 0.0, 103.24), None);
        assert_eq!(ladder_reason(&p, 0, 65.73, 0.0, 103.24), Some(ExitReason::Atr));
        // 創新高後停損上移
        assert_eq!(ladder_reason(&p, 0, 92.0, 0.0, 130.0), Some(ExitReason::Atr));
    }

    // ---- 情境分類 ----
    #[test]
    fn classify_grid() {
        let mut n = 0u64;
        for atr_k in [None, Some(2.5)] { for pnl0 in [0.0, -100.0] { for extra in [0.0, 250.0] { for pnl_big in [300.0, 1000.0] {
            let p = HoldParams { swan_pnl0: pnl0, swan_extra_loss: extra, horizon: 100, pnl_big, atr_k, atr_value: 15.0 };
            for d in [0, 50, 99, 100, 101] { for s in [50.0, 65.0, 66.0, 100.0] { for pnl in [-600.0, -351.0, -350.0, -250.0, -1.0, 0.0, 1.0, 299.0, 300.0, 999.0, 1000.0, 2000.0] {
                for hwm in [s, s + 37.0, s + 37.5, s + 38.0, 103.24] {
                    assert!(props::classify_ok(&p, s, pnl, d, hwm), "d={} s={} pnl={} hwm={}", d, s, pnl, hwm);
                    n += 1;
                }
            }}}
        }}}}
        assert!(n > 5_000);
    }
    #[test]
    fn classify_known_cases() {
        let p = HoldParams { swan_pnl0: 0.0, swan_extra_loss: 250.0, horizon: 100, pnl_big: 300.0, atr_k: Some(2.5), atr_value: 15.0 };
        assert_eq!(classify_hold(&p, 70.0, -300.0, 10, 100.0), 4); // 熔斷優先於 ATR
        assert_eq!(classify_hold(&p, 60.0, -100.0, 10, 103.24), 6); // 60 < 65.74
        assert_eq!(classify_hold(&p, 80.0, -100.0, 10, 103.24), 3); // 時間停損
        assert_eq!(classify_hold(&p, 100.0, 50.0, 10, 103.24), 5); // 區間止盈
        assert_eq!(classify_hold(&p, 150.0, 500.0, 100, 160.0), 0); // 大幅獲利
        assert_eq!(classify_hold(&p, 100.0, 50.0, 100, 103.24), 1);
        assert_eq!(classify_hold(&p, 90.0, -50.0, 100, 103.24), 2);
    }

    // ---- 設定檔 ----
    fn fresh() -> Ctx { trade() }
    #[test]
    fn config_default_ok_and_invariants() {
        let mut c = fresh();
        apply_config(&mut c, DEFAULT_CONFIG, None).expect("內建設定必須有效");
        assert!(c.n_paths >= 1 && c.oos_n >= 1 && c.match_days >= 1);
        assert_eq!(c.iv_init.len(), c.spec.split(';').count());
        assert!(c.actual.windows(2).all(|w| w[0].0 < w[1].0));
        assert!(c.actual.iter().all(|&(d, p)| iso_weekday(d) <= 5 && p > 0.0));
        assert_eq!(c.today, c.actual[0].0);
    }
    #[test]
    fn config_rejects_bad_input_without_panic() {
        let bad = [
            "close = abc 1", "close = 2026-10-20", "close = 2026-10-20 x", "close = 2026-10-20 nan", "close = 2026-10-20 inf",
            "close = 2026-10-20 -5", "close = 2026-10-20 0", "close = 2026-02-30 10", "close = 2026-10-24 10", // 週六
            "close = 2026-10-20 10\nclose = 2026-10-20 11", // 重複
            "iv = 2026-10-23", "iv = 2026-10-23 abc 1", "iv = 2026-13-01 1 1", "iv = 2026-10-23 1.1 -1",
            "n_paths = 0", "n_paths = -5", "n_paths = nan", "n_paths = 1e12", "match_days = 0", "match_days = 1000",
            "debit = 0", "debit = -1", "window_tp = -1", "atr_value = -1", "atr_k = -2", "atr_scan = 1 x", "atr_scan = 0",
            "leg = ", "leg = +1", "leg = +1 125 2027-01-15", "leg = +1 125X 2027-01-15", "leg = 0 125C 2027-01-15",
            "leg = +1 0C 2027-01-15", "leg = +1 125C 2027-02-30", "leg = x 125C 2027-01-15", "leg = +1 C 2027-01-15",
            "unknown_key = 1", "no equals sign", "= 5", "entry_date = 2026-10-21", "time_stop_date = 11/06", "short_term_date =",
            "leg = +1 125C 2027-01-15\nleg = -1 140C 2026-12-18\nleg = +1 100C 2026-12-18", // IV 數量與 leg 不一致
        ];
        for b in bad {
            let mut c = fresh();
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| { apply_config(&mut c, DEFAULT_CONFIG, None).unwrap(); apply_config(&mut c, b, None) }));
            match r { Ok(res) => assert!(res.is_err(), "應回傳 Err: {:?}", b), Err(_) => panic!("不應 panic: {:?}", b) }
        }
    }
    #[test]
    fn config_mutations_never_panic() {
        // 對內建設定做大量變異（逐行刪除、每個位置截斷、替換單一字元）：只能回傳 Ok 或 Err，不能 panic
        let base = DEFAULT_CONFIG;
        let mut variants: Vec<String> = Vec::new();
        for (i, _) in base.lines().enumerate() { variants.push(base.lines().enumerate().filter(|&(j, _)| j != i).map(|(_, l)| l).collect::<Vec<_>>().join("\n")); }
        for (i, _) in base.char_indices() { variants.push(base[..i].to_string()); }
        let reps = ['x', '-', '=', '0', '9', ' ', '#', '\t', '\u{3000}', '億'];
        for (i, ch) in base.char_indices() { for &r in &reps { let mut s = String::new(); s.push_str(&base[..i]); s.push(r); s.push_str(&base[i + ch.len_utf8()..]); variants.push(s); } }
        assert!(variants.len() > 1000);
        for v in &variants {
            let mut c = fresh();
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| { let _ = apply_config(&mut c, v, None); }));
            assert!(res.is_ok(), "panic：{:?}", v);
        }
    }
    #[test]
    fn config_valid_variants_take_effect() {
        let mut c = fresh();
        apply_config(&mut c, DEFAULT_CONFIG, None).unwrap();
        apply_config(&mut c, "close = 2026-10-27 95.5\niv = 2026-10-27 1.18 1.25\nn_paths = 2000\natr_k = 0", None).unwrap();
        assert_eq!(c.actual.last().unwrap().1, 95.5);
        assert_eq!(c.n_paths, 2000);
        assert_eq!(c.iv_init, vec![1.18, 1.25]);
        assert!(c.exit_spec.atr.is_none() && c.cfg.atr_k.is_none());
        // max_closes 截斷：回放用
        let mut c2 = fresh();
        apply_config(&mut c2, DEFAULT_CONFIG, Some(2)).unwrap();
        assert_eq!(c2.actual.len(), 2);
    }

    // ---- 整合：小規模實跑，檢查出場與分類的不變量 ----
    #[test]
    fn small_simulation_invariants() {
        QUIET.store(true, Ordering::Relaxed);
        let mut c = fresh();
        apply_config(&mut c, DEFAULT_CONFIG, None).unwrap();
        c.n_paths = 400;
        setup_trade(&mut c);
        let (results, _) = run(&mut c, &[]);
        let oc = classify_all(&results, &c);
        let nlev = c.outcome.levels().len();
        assert_eq!(results.len(), 400);
        for (r, &o) in results.iter().zip(&oc) {
            assert!(o < nlev);
            assert!(r.dates.windows(2).all(|w| w[0] < w[1]));
            assert_eq!(r.dates.len(), r.s.len());
            assert_eq!(r.dates.len(), r.pnl.len());
            // 已發生的實際收盤日不能出場：路徑至少包含全部實際收盤
            assert!(r.dates.len() >= c.actual.len());
            assert!(*r.dates.last().unwrap() <= c.cfg.horizon);
            assert!(r.s.iter().all(|&x| x.is_finite() && x > 0.0) && r.pnl.iter().all(|x| x.is_finite()));
        }
        for (i, &(d, p)) in c.actual.iter().enumerate() { assert!(results.iter().all(|r| r.dates[i] == d && r.s[i] == p)); }
        QUIET.store(false, Ordering::Relaxed);
    }
}

#[cfg(kani)]
mod verification {
    use super::*;

    fn any_opt() -> Option<f64> {
        let has: bool = kani::any();
        let v: f64 = kani::any();
        kani::assume(v.is_finite() && v.abs() < 1.0e6);
        if has { Some(v) } else { None }
    }
    fn any_f(lim: f64) -> f64 { let v: f64 = kani::any(); kani::assume(v.is_finite() && v.abs() < lim); v }

    // ---------- 日期 ----------
    #[kani::proof]
    fn civil_from_days_is_valid_and_invertible() {
        let z: Day = kani::any();
        kani::assume(z >= -25567 && z <= 110000); // 1900 ~ 2271
        assert!(props::civil_roundtrip(z));
    }
    #[kani::proof]
    fn days_from_civil_then_back() {
        let (y, m, d): (i32, u32, u32) = (kani::any(), kani::any(), kani::any());
        kani::assume(y >= 1900 && y <= 2300 && m >= 1 && m <= 12 && d >= 1 && d <= days_in_month(y, m));
        assert!(props::ymd_roundtrip(y, m, d));
    }
    #[kani::proof]
    fn weekday_in_range_and_advances() {
        let z: Day = kani::any();
        kani::assume(z >= -25567 && z <= 110000);
        assert!(props::weekday_ok(z));
    }
    #[kani::proof]
    #[kani::unwind(12)]
    fn parse_ymd_accepts_only_real_dates() {
        let b: [u8; 10] = kani::any();
        assert!(props::parse_ymd_ok(&b));
    }
    #[kani::proof]
    #[kani::unwind(14)]
    fn parse_ymd_any_length_never_panics() {
        let b: [u8; 12] = kani::any();
        let n: usize = kani::any();
        kani::assume(n <= 12);
        assert!(props::parse_ymd_ok(&b[..n]));
    }

    // ---------- 出場規則 ----------
    #[kani::proof]
    fn ladder_reason_is_consistent_for_all_inputs() {
        let p = LadderParams {
            horizon: kani::any(), stop: any_opt(), ts: kani::any(), ts_date: kani::any(), br: kani::any(), wtp: any_opt(),
            extra: any_f(1.0e6), pnl0: any_f(1.0e6), swan_px: any_f(1.0e6), atr_d: any_opt(),
        };
        let (d, s, pnl, hwm): (Day, f64, f64, f64) = (kani::any(), any_f(1.0e6), any_f(1.0e6), any_f(1.0e6));
        assert!(props::ladder_consistent(&p, d, s, pnl, hwm));
    }
    #[kani::proof]
    fn window_take_profit_semantics() {
        let (ts_date, horizon, d): (Day, Day, Day) = (kani::any(), kani::any(), kani::any());
        let (cap, pnl) = (any_f(1.0e6), any_f(1.0e6));
        kani::assume(cap >= 0.0 && ts_date <= d && d < horizon);
        assert!(props::window_semantics(ts_date, horizon, cap, d, pnl));
    }
    #[kani::proof]
    fn nothing_fires_before_time_stop_date() {
        let (ts_date, horizon, d): (Day, Day, Day) = (kani::any(), kani::any(), kani::any());
        let (cap, pnl) = (any_f(1.0e6), any_f(1.0e6));
        kani::assume(cap >= 0.0 && d < ts_date && d < horizon);
        assert!(props::before_time_stop_quiet(ts_date, horizon, cap, d, pnl));
    }
    #[kani::proof]
    fn atr_never_fires_at_new_high() {
        let (a, s) = (any_f(1.0e6), any_f(1.0e6));
        kani::assume(a >= 0.0);
        assert!(props::atr_not_at_new_high(a, s));
    }

    // ---------- 情境分類 ----------
    #[kani::proof]
    fn classify_hold_total_and_consistent() {
        let p = HoldParams {
            swan_pnl0: any_f(1.0e6), swan_extra_loss: any_f(1.0e6), horizon: kani::any(), pnl_big: any_f(1.0e6),
            atr_k: any_opt(), atr_value: any_f(1.0e6),
        };
        let (s_end, pnl_end, d_end, hwm): (f64, f64, Day, f64) = (any_f(1.0e6), any_f(1.0e6), kani::any(), any_f(1.0e6));
        assert!(props::classify_ok(&p, s_end, pnl_end, d_end, hwm));
    }
}
