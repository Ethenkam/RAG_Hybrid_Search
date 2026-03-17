import os
import sys
import subprocess
import threading
import time
import shutil

def main():
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        try:
            api_key = input("Введите ваш MISTRAL_API_KEY: ").strip()
        except EOFError:
            pass

    if not api_key:
        print("❌ Ключ не введён. Завершение.")
        sys.exit(1)

    os.environ["MISTRAL_API_KEY"] = api_key

    # Импорты FastAPI (должны быть установлены)
    try:
        from app import app
        import uvicorn
        import nest_asyncio
    except ImportError:
        print("❌ Ошибка: Не установлены библиотеки. Выполните: pip install fastapi uvicorn nest_asyncio")
        sys.exit(1)

    nest_asyncio.apply()

    def run_fastapi():
        # log_level="error" чтобы не засорять консоль логами uvicorn
        host = os.environ.get("HOST", "0.0.0.0")
        port = int(os.environ.get("PORT", "8000"))
        uvicorn.run(app, host=host, port=port, log_level="error")

    server_thread = threading.Thread(target=run_fastapi, daemon=True)
    server_thread.start()

    print("⏳ Ожидание запуска сервера (3 сек)...")
    time.sleep(3)

    lt_process = None

    # Проверяем, есть ли вообще npx в системе
    if not shutil.which("npx"):
        print("❌ Ошибка: 'npx' не найден.")
        print("👉 Установите Node.js командой: sudo apt install nodejs npm")

    try:
        print("🚀 Запуск localtunnel...")

        # ВАЖНО: флаг "-y" автоматически соглашается на установку пакета
        command = ["npx", "-y", "localtunnel", "--port", "8000"]

        lt_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, # Ловим ошибки
            text=True,
            bufsize=1 # Построчный буфер
        )

        print("🔍 Ожидание URL...")
        
        while True:
            # Читаем вывод построчно
            line = lt_process.stdout.readline()
            if not line and lt_process.poll() is not None:
                break
            
            if line:
                clean_line = line.strip()
                # localtunnel иногда выводит мусор, фильтруем
                if "your url is" in clean_line.lower():
                    url = clean_line.split("is")[-1].strip()
                    print(f"\n✅ \033[92mПубличный URL: {url}\033[0m")
                    print("🌍 (Нажмите Ctrl+C, чтобы остановить сервер)\n")
                elif "error" in clean_line.lower():
                    print(f"⚠️ LT Error: {clean_line}")

            # Если процесс упал, читаем ошибку
            if lt_process.poll() is not None:
                err = lt_process.stderr.read()
                print(f"❌ Localtunnel упал с ошибкой:\n{err}")
                break

    except KeyboardInterrupt:
        print("\n🛑 Останавливаем...")
    except Exception as e:
        print(f"⚠️ Критическая ошибка: {e}")

    finally:
        if lt_process:
            lt_process.terminate()
            try:
                lt_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                lt_process.kill()
        print("👋 Программа завершена.")

if __name__ == "__main__":
    main()