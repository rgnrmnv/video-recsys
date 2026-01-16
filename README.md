## Рекомендательная система

FastAPI‑сервис, который отдает персональные рекомендации коротких видео на основе алгоритма схожести эмбеддингов.

### Быстрый старт (локально)
- Установить Python 3.9+.
- Установить зависимости: `pip install -r requirements.txt`
- Запустить API: `uvicorn app.main:app --reload --port 8000`
- Проверка: `curl http://localhost:8000/health`

### Запуск Docker
- Сборка образа: `docker build -t video-recommender .`
- Запуск контейнера: `docker run --rm -p 8000:8000 video-recommender`
- Проверка health: `curl http://localhost:8000/health`

### API
- `GET /health` → `{"status": "ok"}`
- `POST /recommend`
  - Тело запроса:
    ```json
    {
      "user_id": "username",
      "watched_videos": ["v1", "v2"],
      "liked_categories": ["education", "science"]
    }
    ```
  - Пример curl:
    ```bash
    curl -X POST http://localhost:8000/recommend \
      -H "Content-Type: application/json" \
      -d '{"user_id":"username","watched_videos":["v1","v2","v6"],"liked_categories":["education","science"]}'
    ```
  - Пример ответа:
    ```json
    {
      "user_id": "user-123",
      "recommendations": ["v3", "v5", "v4"],
      "algorithm_version": "1.0"
    }
    ```
- Сервис слушает порт `8000`

### ML‑алгоритм
- Загружает `videos_data.csv` с заранее посчитанными эмбеддингами.
- Строит центроиды категорий (среднее эмбеддингов по каждой категории).
- Оценивает видео (кроме уже просмотренных) по косинусному сходству с центроидами любимых категорий; если нет совпадений, используется глобальный усредненный вектор.
- Возвращает топ‑5 видео по убыванию сходства. Версия алгоритма: `1.0`.

### Структура проекта
- `app/main.py` : FastAPI, эндпоинты и логика рекомендаций.
- `videos_data.csv` : каталог с эмбеддингами.
- `Dockerfile`, `requirements.txt` : контейнеризация и зависимости.
