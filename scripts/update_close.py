import re, sys
import yfinance as yf

txt = open("config.txt", encoding="utf-8").read()
entry = re.search(r"^entry_date\s*=\s*(\S+)", txt, re.M).group(1)
label = re.search(r"^label\s*=\s*(\S+)", txt, re.M).group(1)

df = yf.download(label, start=entry, auto_adjust=False, progress=False)
closes = df["Close"].squeeze().dropna()
if closes.empty:
    sys.exit("抓不到資料，config.txt 不更新")

lines = [f"close = {d.strftime('%Y-%m-%d')} {v:.2f}" for d, v in closes.items()]
if not lines[0].startswith(f"close = {entry} "):
    sys.exit(f"第一筆收盤 {lines[0]} 與 entry_date {entry} 不符，請檢查")

rest = [l for l in txt.splitlines() if not l.strip().startswith("close")]
i = next((k for k, l in enumerate(rest) if "實際收盤" in l), None)
if i is None:
    i = next(k for k, l in enumerate(rest) if l.startswith("leg"))
open("config.txt", "w", encoding="utf-8").write(
    "\n".join(rest[: i + 1] + lines + rest[i + 1 :]) + "\n"
)
print("\n".join(lines[-3:]))
