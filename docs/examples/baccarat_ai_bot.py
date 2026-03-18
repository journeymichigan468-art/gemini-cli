"""Telegram Baccarat prediction bot example.

This script combines:
- A TensorFlow model prediction
- A lightweight in-memory pattern learner saved to disk
- Simple streak/chop/bias heuristics

Environment variables:
- BACCARAT_BOT_TOKEN: Telegram bot token
Optional overrides:
- BACCARAT_MODEL_PATH
- BACCARAT_MEMORY_FILE
"""

from __future__ import annotations

import json
import math
import os
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import telebot
from tensorflow.keras.models import load_model

# =====================================
# CONFIGURATION
# =====================================

API_TOKEN = os.getenv("BACCARAT_BOT_TOKEN")
if not API_TOKEN:
    raise ValueError("Set BACCARAT_BOT_TOKEN environment variable")

MODEL_PATH = os.getenv("BACCARAT_MODEL_PATH", "/root/baccarat_model/baccarat_model.h5")
MEMORY_FILE = os.getenv("BACCARAT_MEMORY_FILE", "/root/baccarat_model/memory.json")

bot = telebot.TeleBot(API_TOKEN)
history: List[str] = []
memory_lock = threading.Lock()

# =====================================
# LOAD MODEL SAFELY
# =====================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = load_model(MODEL_PATH)

# =====================================
# MEMORY SYSTEM
# =====================================

if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        memory: Dict[str, Dict[str, Dict[str, int]]] = json.load(f)
else:
    memory = {}


def save_memory() -> None:
    with memory_lock:
        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2)


def learn_from_outcome(previous_history: List[str], outcome: str) -> None:
    """Learn using history BEFORE outcome."""
    with memory_lock:
        for length in range(2, 13):
            if len(previous_history) >= length:
                pattern = "".join(previous_history[-length:])
                length_key = str(length)

                if length_key not in memory:
                    memory[length_key] = {}

                if pattern not in memory[length_key]:
                    memory[length_key][pattern] = {"P": 0, "B": 0, "T": 0}

                memory[length_key][pattern][outcome] += 1

    save_memory()


def calculate_adjusted_confidence(
    counts: Dict[str, int], total_occurrences: int, k: int = 20
) -> Tuple[float, Optional[str]]:
    total = sum(counts.values())
    if total == 0 or total_occurrences < 3:
        return 0.0, None

    best = max(counts, key=counts.get)
    raw_conf = counts[best] / total
    adjusted_conf = raw_conf * (1 - math.exp(-total_occurrences / k))

    return adjusted_conf, best


def get_memory_prediction(seq: List[str]) -> Tuple[float, Optional[str]]:
    length = len(seq)
    pattern = "".join(seq)

    with memory_lock:
        if str(length) in memory and pattern in memory[str(length)]:
            counts = memory[str(length)][pattern]
            total_occurrences = sum(counts.values())
            return calculate_adjusted_confidence(counts, total_occurrences)

    return 0.0, None


# =====================================
# PATTERN DETECTORS
# =====================================


def detect_streak(hist: List[str], min_streak: int = 3) -> Tuple[int, Optional[str]]:
    if len(hist) < min_streak:
        return 0, None

    last = hist[-1]
    streak = 1

    for i in range(len(hist) - 2, -1, -1):
        if hist[i] == last:
            streak += 1
        else:
            break

    if streak >= min_streak:
        return streak, last

    return 0, None


def detect_chop(hist: List[str]) -> bool:
    if len(hist) < 5:
        return False

    last5 = hist[-5:]
    return last5 in [["P", "B", "P", "B", "P"], ["B", "P", "B", "P", "B"]]


def detect_bias(hist: List[str], window: int = 20) -> int:
    if len(hist) < window:
        return 0

    recent = hist[-window:]
    p = recent.count("P")
    b = recent.count("B")

    if p + b == 0:
        return 0

    ratio = p / (p + b)

    if ratio > 0.6:
        return 1
    if ratio < 0.4:
        return -1
    return 0


def detect_pattern_strength(hist: List[str]) -> int:
    score = 0

    streak_len, streak_type = detect_streak(hist)
    if streak_type == "P":
        score += streak_len
    elif streak_type == "B":
        score -= streak_len

    if detect_chop(hist):
        score += -2 if hist[-1] == "P" else 2

    score += detect_bias(hist) * 3

    return score


# =====================================
# ML PREDICTION
# =====================================


def ml_predict(seq: List[str]) -> str:
    seq_int = [0 if x == "P" else 1 if x == "B" else 2 for x in seq]
    seq_array = np.array(seq_int).reshape(1, len(seq), 1)

    try:
        pred = model.predict(seq_array, verbose=0)[0]
    except Exception as exc:  # Model/runtime fallback
        print("Model prediction error:", exc)
        return str(np.random.choice(["P", "B"]))

    lstm_choice = int(np.argmax(pred))

    lstm_score = 2 if lstm_choice == 0 else -2 if lstm_choice == 1 else 0
    pattern_score = detect_pattern_strength(history)

    memory_conf, memory_pred = get_memory_prediction(seq)
    memory_score = 0

    if memory_pred == "P":
        memory_score = int(memory_conf * 5)
    elif memory_pred == "B":
        memory_score = -int(memory_conf * 5)

    total = lstm_score + pattern_score + memory_score

    if total > 1:
        return "P"
    if total < -1:
        return "B"
    return ["P", "B", "T"][lstm_choice]


# =====================================
# TELEGRAM HANDLERS
# =====================================


@bot.message_handler(commands=["start"])
def start(message):
    bot.reply_to(
        message,
        "🎰 Baccarat AI Bot Ready!\n"
        "Send P / B / T\n"
        "/learn P - teach real result",
    )


@bot.message_handler(commands=["learn"])
def learn(message):
    global history

    try:
        outcome = message.text.split()[1].upper()
    except (AttributeError, IndexError):
        bot.reply_to(message, "Usage: /learn P")
        return

    if outcome not in ["P", "B", "T"] or len(history) < 2:
        bot.reply_to(message, "Usage: /learn P")
        return

    previous = history.copy()
    learn_from_outcome(previous, outcome)

    bot.reply_to(message, "✔️ Learned successfully")


@bot.message_handler(func=lambda m: True)
def predict(message):
    global history

    val = (message.text or "").upper()

    if val not in ["P", "B", "T"]:
        bot.reply_to(message, "Send only P / B / T")
        return

    history.append(val)

    if len(history) >= 9:
        prediction = ml_predict(history[-9:])
    else:
        bot.reply_to(message, f"Need {9 - len(history)} more inputs")
        return

    last5 = "-".join(history[-5:])
    bot.reply_to(message, f"History: {last5}\nPrediction: {prediction}")


# =====================================
# START BOT
# =====================================

print("Bot running...")
bot.infinity_polling(timeout=60, long_polling_timeout=60)
