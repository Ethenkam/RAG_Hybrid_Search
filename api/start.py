import os
import sys
import subprocess
import threading
import time

def main():
    api_key = input("Введите ваш MISTRAL_API_KEY: ").strip()
    if not api_key:
        print("❌ Ключ не введён. Завершение.")
        sys.exit(1)

    os.environ["MISTRAL_API_KEY"] = api_key

    from app import app
    import uvicorn
    import nest_asyncio

    nest_asyncio.apply()

    def run_fastapi():
        uvicorn.run(app, host="127.0.0.1", port=8000)

    server_thread = threading.Thread(target=run_fastapi, daemon=True)
    server_thread.start()

    print("⏳ Ожидание запуска сервера...")
    time.sleep(3)  # чуть больше времени для тяжёлых моделей

    lt_process = None  # ← ИНИЦИАЛИЗИРУЕМ ЗАРАНЕЕ

    try:
        print("🚀 Запуск localtunnel...")
        lt_process = subprocess.Popen(
            ["npx", "localtunnel", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in iter(lt_process.stdout.readline, ""):
            print("lt:", line.strip())
            if "https://" in line and ".loca.lt" in line:
                public_url = line.strip().split()[-1]
                print(f"\n✅ Публичный URL: {public_url}")
                print("Оставьте это окно открытым для работы туннеля.\n")

    except FileNotFoundError:
        print("⚠️ localtunnel не найден. Установите Node.js и выполните вручную:")
        print("   npx localtunnel --port 8000")
    except KeyboardInterrupt:
        print("\n🛑 Завершение...")
        sys.exit(0)
    except Exception as e:
        print(f"⚠️ Ошибка при запуске localtunnel: {e}")

    print("FastAPI работает. Нажмите Ctrl+C для выхода.")
    try:
        if lt_process:
            lt_process.wait()  # ← только если запущен
        else:
            # Если localtunnel не запущен — просто ждём вручную
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        if lt_process:
            lt_process.terminate()
            lt_process.wait()
        print("\n✅ Сервер остановлен.")

if __name__ == "__main__":
    main()