# Настройка Docker для Intel Arc B580

Этот проект можно запустить внутри Docker-контейнера, настроенного для графических процессоров Intel Arc (XPU). Используется официальный образ `intel/intel-extension-for-pytorch:2.7.10-xpu`.

## Требования

1.  **Драйверы Intel GPU**: Убедитесь, что на хост-машине установлены актуальные драйверы Intel Arc.
    *   Ubuntu 24.04: `sudo apt install intel-opencl-icd intel-level-zero-gpu level-zero`
    *   Windows (WSL2): Установите драйверы для Windows; WSL2 обычно пробрасывает их автоматически.
2.  **Docker**: Установите Docker и Docker Compose.
3.  **Ключ API Mistral**: Вам нужен API-ключ от [Mistral AI](https://console.mistral.ai/).

## Быстрый старт

Запустите вспомогательный скрипт:

```bash
chmod +x start_docker.sh
./start_docker.sh
```

Скрипт выполнит:
1.  Проверку наличия `MISTRAL_API_KEY` (запросит, если не найден).
2.  Сборку Docker-образа.
3.  Запуск контейнера с доступом к GPU.

## Ручной запуск

Если вы предпочитаете запускать через `docker-compose` напрямую:

1.  Задайте ключ API:
    ```bash
    export MISTRAL_API_KEY=your_key_here
    ```
2.  Запустите:
    ```bash
    docker-compose up --build
    ```

## Структура файлов

*   `Dockerfile`: Собирает образ на базе Intel XPU-образа.
*   `docker-compose.yml`: Описывает сервис, проброс GPU (`/dev/dri`) и монтирование томов.
*   `.dockerignore`: Исключает ненужные файлы из контекста сборки.

## Устранение неполадок

*   **Permission Denied `/dev/dri`**: Убедитесь, что ваш пользователь входит в группу `render`: `sudo usermod -aG render $USER`.
*   **Отсутствующие зависимости**: Если появляется `ModuleNotFoundError`, проверьте логику фильтрации `requirements.txt` в `Dockerfile`.