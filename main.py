import asyncio
import os
import logging
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from huggingface_hub import AsyncInferenceClient
from dotenv import load_dotenv

# Загружаем переменные из .env (для локального запуска)
load_dotenv()

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Получение токенов ---
TG_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

if not TG_BOT_TOKEN or not HF_TOKEN:
    raise ValueError("Не найдены токены! Проверь переменные окружения.")

# --- Инициализация ---
bot = Bot(token=TG_BOT_TOKEN)
dp = Dispatcher()

# Hugging Face клиент (асинхронный)
# Модель можно заменить на любую доступную: meta-llama/Llama-3.2-3B-Instruct, mistralai/Mistral-7B-Instruct-v0.3 и т.д.
hf_client = AsyncInferenceClient(
    api_key=HF_TOKEN,
    model="meta-llama/Meta-Llama-3-8B-Instruct"
)

# --- Обработчики ---
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Привет! Я бот с ИИ (Hugging Face). Напиши мне что-нибудь!"
    )

@dp.message(F.text)
async def handle_text(message: types.Message):
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")

    try:
        # Асинхронный запрос к Hugging Face
        response = await hf_client.chat_completion(
            messages=[
                {"role": "system", "content": "Ты полезный ассистент. Отвечай кратко и по делу."},
                {"role": "user", "content": message.text}
            ],
            max_tokens=500,
            temperature=0.7
        )
        answer = response.choices[0].message.content

        # Telegram ограничивает длину сообщения — разбиваем при необходимости
        if len(answer) > 4000:
            for i in range(0, len(answer), 4000):
                await message.answer(answer[i:i + 4000])
        else:
            await message.answer(answer)

    except Exception as e:
        logger.error(f"Ошибка при запросе к Hugging Face: {e}")
        await message.answer("⚠️ Произошла ошибка. Попробуй позже.")

# --- Запуск ---
async def main():
    logger.info("Бот запускается...")
    # Сбрасываем старые зависшие сообщения, чтобы бот не отвечал на них после перезапуска
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())