from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime, timezone
import os

OUTPUT_DIR = r"C:\Users\wns5255\rehab_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "rehab_stream.csv")

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        ln = int(self.headers.get('Content-Length', '0') or 0)
        body = self.rfile.read(ln).decode('utf-8', errors='ignore').strip()
        if not body:
            self.send_response(400)
            self.end_headers()
            return

        # CSV 라인이 여러 줄일 수도 있으니 줄 단위로 처리
        lines = body.splitlines()
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            for line in lines:
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        epoch_ms = int(parts[0])
                        # epoch_ms → 로컬 시간 문자열
                        dt = datetime.fromtimestamp(epoch_ms/1000, tz=timezone.utc).astimezone()
                        local_time = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # 밀리초까지
                        # 새로운 라인: localTime, epochMs, 나머지 값들
                        new_line = ",".join([local_time, str(epoch_ms)] + parts[1:])
                        f.write(new_line + "\n")
                    except Exception:
                        # epoch 변환 실패하면 원본 그대로 저장
                        f.write(line + "\n")
                else:
                    f.write(line + "\n")

        self.send_response(200)
        self.end_headers()

if __name__ == '__main__':
    PORT = 5555
    print(f"Server listening on 0.0.0.0:{PORT}")
    print(f"Saving to {OUTPUT_FILE}")
    HTTPServer(('0.0.0.0', PORT), Handler).serve_forever()
