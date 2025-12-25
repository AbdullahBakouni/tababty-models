"""
Flask API with BETTER Translation Models + Redis Cache
Uses high-quality models for both directions with Redis caching
"""

from dotenv import load_dotenv

load_dotenv()

import os
import re
import time
import warnings
from functools import lru_cache
from typing import List

from flask import Flask, jsonify, request
from flask_caching import Cache

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Redis Cache configuration
# REDIS_HOST = os.getenv("REDIS_HOST", "redis")
# REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
# REDIS_DB = int(os.getenv("REDIS_DB", 0))
# CACHE_DEFAULT_TIMEOUT = int(os.getenv("CACHE_DEFAULT_TIMEOUT", 3600))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_DEFAULT_TIMEOUT = int(os.getenv("CACHE_DEFAULT_TIMEOUT", 3600))
# cache_config = {
#     "CACHE_TYPE": "RedisCache",
#     "CACHE_REDIS_HOST": REDIS_HOST,
#     "CACHE_REDIS_PORT": REDIS_PORT,
#     "CACHE_REDIS_DB": REDIS_DB,
#     "CACHE_DEFAULT_TIMEOUT": CACHE_DEFAULT_TIMEOUT,
#     "CACHE_KEY_PREFIX": "transliteration:",
# }

# print(f"📦 Redis Configuration:")
# print(f"   Host: {REDIS_HOST}")
# print(f"   Port: {REDIS_PORT}")
# print(f"   DB: {REDIS_DB}")
# print(f"   Timeout: {CACHE_DEFAULT_TIMEOUT}s")

cache_config = {
    "CACHE_TYPE": "RedisCache",
    "CACHE_REDIS_URL": REDIS_URL,  # MUST be rediss://
    "CACHE_DEFAULT_TIMEOUT": CACHE_DEFAULT_TIMEOUT,
    "CACHE_KEY_PREFIX": "transliteration:",
}

print(f"📦 Redis Configuration:")
print(f"   URL: {REDIS_URL.split('@')[-1]}")  # Print only the endpoint for security
print(f"   Timeout: {CACHE_DEFAULT_TIMEOUT}s")

# try:
#     cache = Cache(app, config=cache_config)
#     print("✅ Redis cache initialized successfully")
# except Exception as e:
#     print(f"⚠️  Redis connection failed: {e}")
#     print("   Falling back to SimpleCache")
#     cache_config = {
#         "CACHE_TYPE": "SimpleCache",
#         "CACHE_DEFAULT_TIMEOUT": CACHE_DEFAULT_TIMEOUT,
#         "CACHE_THRESHOLD": 10000,
#     }
#     cache = Cache(app, config=cache_config)
try:
    cache = Cache(app, config=cache_config)
    with app.app_context():
        cache.set("ping", "pong", timeout=10)
        if cache.get("ping") == "pong":
            print("✅ Upstash Redis connected successfully")
        else:
            raise Exception("Ping test failed")
except Exception as e:
    print(f"⚠️ Redis connection failed: {e}")
    print("   Falling back to SimpleCache")
    cache = Cache(
        app,
        config={
            "CACHE_TYPE": "SimpleCache",
            "CACHE_DEFAULT_TIMEOUT": CACHE_DEFAULT_TIMEOUT,
            "CACHE_THRESHOLD": 10000,
        },
    )

# Global variables
ar_en_model = None
en_ar_model = None
ar_en_tokenizer = None
en_ar_tokenizer = None
device = None
USE_TRANSFORMERS = True

# Which models to use
MODEL_CHOICE = os.getenv("MODEL_CHOICE", "opus-big")  # Options: "opus-big", "marefa"

# Performance metrics
metrics = {
    "cache_hits": 0,
    "cache_misses": 0,
    "model_requests": 0,
    "total_requests": 0,
    "avg_response_time": 0.0,
}

print("🚀 Starting Transliteration API with Better Models + Redis Cache...")


def load_models():
    """Load BETTER translation models"""
    global \
        ar_en_model, \
        en_ar_model, \
        ar_en_tokenizer, \
        en_ar_tokenizer, \
        device, \
        USE_TRANSFORMERS, \
        MODEL_CHOICE

    try:
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {device}")

        if MODEL_CHOICE == "opus-big":
            # OPTION 1: OPUS Big Models (BEST QUALITY - Recommended)
            print("📥 Loading OPUS-MT-BIG models (high quality)...")

            # Arabic to English (OPUS Big)
            print("   Loading AR→EN (opus-mt-tc-big-ar-en)...")
            ar_en_model_name = "Helsinki-NLP/opus-mt-tc-big-ar-en"
            ar_en_tokenizer = MarianTokenizer.from_pretrained(ar_en_model_name)
            ar_en_model = MarianMTModel.from_pretrained(ar_en_model_name).to(device)
            ar_en_model.eval()
            print("   ✅ AR→EN loaded")

            # English to Arabic (OPUS Big) - MUCH BETTER than basic model
            print("   Loading EN→AR (opus-mt-tc-big-en-ar)...")
            en_ar_model_name = "Helsinki-NLP/opus-mt-tc-big-en-ar"
            en_ar_tokenizer = MarianTokenizer.from_pretrained(en_ar_model_name)
            en_ar_model = MarianMTModel.from_pretrained(en_ar_model_name).to(device)
            en_ar_model.eval()
            print("   ✅ EN→AR loaded")

            print("🎉 OPUS-MT-BIG models loaded successfully!")

        elif MODEL_CHOICE == "marefa":
            # OPTION 2: Marefa Model (Specialized for Arabic)
            print("📥 Loading Marefa models (Arabic-specialized)...")

            # Arabic to English (OPUS Big - still best for this direction)
            print("   Loading AR→EN (opus-mt-tc-big-ar-en)...")
            ar_en_model_name = "Helsinki-NLP/opus-mt-tc-big-ar-en"
            ar_en_tokenizer = MarianTokenizer.from_pretrained(ar_en_model_name)
            ar_en_model = MarianMTModel.from_pretrained(ar_en_model_name).to(device)
            ar_en_model.eval()
            print("   ✅ AR→EN loaded")

            # English to Arabic (Marefa - Arabic specialized)
            print("   Loading EN→AR (marefa-mt-en-ar)...")
            en_ar_model_name = "marefa-nlp/marefa-mt-en-ar"
            en_ar_tokenizer = MarianTokenizer.from_pretrained(en_ar_model_name)
            en_ar_model = MarianMTModel.from_pretrained(en_ar_model_name).to(device)
            en_ar_model.eval()
            print("   ✅ EN→AR loaded (Marefa)")

            print("🎉 Marefa models loaded successfully!")

        USE_TRANSFORMERS = True
        return True

    except Exception as e:
        print(f"⚠️  Error loading models: {str(e)}")
        print("💡 Make sure you have enough memory and internet connection")
        USE_TRANSFORMERS = False
        return False


@lru_cache(maxsize=10000)
def normalize_arabic(text: str) -> str:
    """Normalize Arabic text (cached in memory)"""
    text = re.sub(r"[\u064B-\u065F]", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")
    return text.strip()


@lru_cache(maxsize=10000)
def get_phonetic_variants(text: str) -> List[str]:
    """Get phonetic variants (cached in memory)"""
    variants = {text.lower()}
    base = text.lower()

    patterns = [
        (r"a", "e"),
        (r"e", "a"),
        (r"een$", "ain"),
        (r"ain$", "een"),
        (r"(.)\1", r"\1"),
    ]

    for pattern, replacement in patterns:
        if re.search(pattern, base):
            variant = re.sub(pattern, replacement, base)
            if variant != base and len(variant) >= 2:
                variants.add(variant)

    return list(variants)[:5]


def transliterate_with_model(text: str, from_lang: str, to_lang: str) -> List[str]:
    """Use high-quality transformer models"""
    global ar_en_model, en_ar_model, ar_en_tokenizer, en_ar_tokenizer, device

    if not USE_TRANSFORMERS:
        return []

    try:
        import torch

        # Select model and tokenizer
        if from_lang == "ar" and to_lang == "en":
            model = ar_en_model
            tokenizer = ar_en_tokenizer
        elif from_lang == "en" and to_lang == "ar":
            model = en_ar_model
            tokenizer = en_ar_tokenizer

            # For OPUS Big EN→AR, need to add language token
            if MODEL_CHOICE == "opus-big":
                text = ">>ara<< " + text
        else:
            return []

        # Tokenize
        inputs = tokenizer([text], return_tensors="pt", padding=True).to(device)
        variants = set()

        with torch.no_grad():
            # Method 1: Greedy decoding
            outputs = model.generate(
                **inputs, max_length=50, num_beams=1, do_sample=False
            )
            result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            if result:
                variants.add(result.lower() if to_lang == "en" else result)

            # Method 2: Beam search (best quality)
            outputs = model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                num_return_sequences=3,
                do_sample=False,
                early_stopping=True,
            )
            for output in outputs:
                result = tokenizer.decode(output, skip_special_tokens=True).strip()
                if result:
                    variants.add(result.lower() if to_lang == "en" else result)

            # Method 3: Diverse beam search
            outputs = model.generate(
                **inputs,
                max_length=50,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
                num_return_sequences=2,
            )
            for output in outputs:
                result = tokenizer.decode(output, skip_special_tokens=True).strip()
                if result:
                    variants.add(result.lower() if to_lang == "en" else result)

        return list(variants)[:6]

    except Exception as e:
        print(f"Model error: {str(e)}")
        return []


def add_arabic_variants(text: str) -> List[str]:
    """Add common Arabic variants"""
    variants = {text}

    # Hamza variants
    if text.startswith("ا"):
        variants.add("أ" + text[1:])
        variants.add("إ" + text[1:])

    # Final ya
    if text.endswith("ي"):
        variants.add(text[:-1] + "ى")

    # Taa marbuta
    if text.endswith("ه"):
        variants.add(text[:-1] + "ة")

    return list(variants)


# ============================================================================
# API ENDPOINTS
# ============================================================================


@app.route("/health", methods=["GET"])
def health_check():
    """Health check with Redis status"""
    redis_status = "connected"
    try:
        # Test Redis connection
        cache.set("health_check", "ok", timeout=5)
        test_val = cache.get("health_check")
        if test_val != "ok":
            redis_status = "error"
    except Exception as e:
        redis_status = f"error: {str(e)}"

    return jsonify(
        {
            "status": "healthy",
            "models_loaded": ar_en_model is not None and en_ar_model is not None,
            "model_type": MODEL_CHOICE,
            "using_transformers": USE_TRANSFORMERS,
            "device": str(device) if device else "none",
            "ar_en_model": "opus-mt-tc-big-ar-en",
            "en_ar_model": "opus-mt-tc-big-en-ar"
            if MODEL_CHOICE == "opus-big"
            else "marefa-mt-en-ar",
            "cache": {
                "type": "Redis",
                "status": redis_status,
                "timeout": CACHE_DEFAULT_TIMEOUT,
            },
            "metrics": metrics,
        }
    )


@app.route("/stats", methods=["GET"])
def get_stats():
    """Performance statistics"""
    cache_total = metrics["cache_hits"] + metrics["cache_misses"]
    hit_rate = metrics["cache_hits"] / cache_total if cache_total > 0 else 0

    # Get Redis info if available
    redis_info = {}
    try:
        if cache_config.get("CACHE_TYPE") == "RedisCache":
            redis_client = cache.cache._write_client
            info = redis_client.info("stats")
            redis_info = {
                "total_connections_received": info.get("total_connections_received", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
            }
    except Exception as e:
        redis_info = {"error": str(e)}

    return jsonify(
        {
            "cache": {
                "hits": metrics["cache_hits"],
                "misses": metrics["cache_misses"],
                "hit_rate": f"{hit_rate * 100:.2f}%",
                "redis": redis_info,
            },
            "requests": {
                "total": metrics["total_requests"],
                "model_requests": metrics["model_requests"],
            },
            "performance": {
                "avg_response_time_ms": f"{metrics['avg_response_time']:.2f}",
            },
            "model_info": {
                "type": MODEL_CHOICE,
                "ar_en": "opus-mt-tc-big-ar-en",
                "en_ar": "opus-mt-tc-big-en-ar"
                if MODEL_CHOICE == "opus-big"
                else "marefa-mt-en-ar",
            },
        }
    )


@app.route("/transliterate", methods=["POST"])
def transliterate():
    """Main transliteration endpoint with Redis caching"""
    start_time = time.time()
    metrics["total_requests"] += 1

    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        from_lang = data.get("from", "en")
        to_lang = data.get("to", "ar")

        if not text or len(text) < 2:
            return jsonify({"variants": []})

        # Redis cache key
        cache_key = f"{from_lang}-{to_lang}-{text}"

        # Check Redis cache
        cached = cache.get(cache_key)
        if cached:
            metrics["cache_hits"] += 1
            return jsonify(
                {"variants": cached, "cached": True, "cache_source": "redis"}
            )

        metrics["cache_misses"] += 1
        metrics["model_requests"] += 1

        # Generate variants with model
        variants = set([text, text.lower()])

        if from_lang == "ar":
            # Normalize Arabic
            normalized = normalize_arabic(text)
            if normalized != text:
                variants.add(normalized)

            # Use model
            model_variants = transliterate_with_model(normalized, from_lang, to_lang)
            variants.update(model_variants)

            # Add phonetic variants
            for v in list(variants):
                if re.match(r"^[a-z]+$", v):
                    variants.update(get_phonetic_variants(v))

        else:  # English to Arabic
            # Use model
            model_variants = transliterate_with_model(text, from_lang, to_lang)
            variants.update(model_variants)

            # Add Arabic variants for each result
            for v in list(variants):
                if re.search(r"[\u0600-\u06FF]", v):
                    variants.update(add_arabic_variants(v))

        # Filter and limit
        result = [v for v in variants if v and len(v) >= 2][:6]

        # Cache result in Redis
        cache.set(cache_key, result, timeout=CACHE_DEFAULT_TIMEOUT)

        # Update metrics
        response_time = (time.time() - start_time) * 1000
        metrics["avg_response_time"] = (
            metrics["avg_response_time"] * (metrics["total_requests"] - 1)
            + response_time
        ) / metrics["total_requests"]

        return jsonify(
            {
                "variants": result,
                "cached": False,
                "response_time_ms": round(response_time, 2),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e), "variants": []}), 500


@app.route("/transliterate/batch", methods=["POST"])
def transliterate_batch():
    """Batch transliteration with Redis caching"""
    start_time = time.time()

    try:
        data = request.get_json()
        texts = data.get("texts", [])
        from_lang = data.get("from", "en")
        to_lang = data.get("to", "ar")

        results = {}
        cache_hits = 0
        cache_misses = 0

        for text in texts:
            cache_key = f"{from_lang}-{to_lang}-{text}"
            cached = cache.get(cache_key)

            if cached:
                results[text] = cached
                metrics["cache_hits"] += 1
                cache_hits += 1
            else:
                metrics["cache_misses"] += 1
                cache_misses += 1
                variants = set([text, text.lower()])

                # Use models
                model_variants = transliterate_with_model(text, from_lang, to_lang)
                variants.update(model_variants)

                # Add variants
                if to_lang == "ar":
                    for v in list(variants):
                        if re.search(r"[\u0600-\u06FF]", v):
                            variants.update(add_arabic_variants(v))
                else:
                    for v in list(variants):
                        if re.match(r"^[a-z]+$", v):
                            variants.update(get_phonetic_variants(v))

                result = [v for v in variants if v and len(v) >= 2][:6]
                results[text] = result

                # Cache in Redis
                cache.set(cache_key, result, timeout=CACHE_DEFAULT_TIMEOUT)

        metrics["total_requests"] += len(texts)
        response_time = (time.time() - start_time) * 1000

        return jsonify(
            {
                "results": results,
                "count": len(results),
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "response_time_ms": round(response_time, 2),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e), "results": {}}), 500


@app.route("/cache/clear", methods=["POST"])
def clear_cache():
    """Clear Redis cache"""
    try:
        cache.clear()
        metrics["cache_hits"] = 0
        metrics["cache_misses"] = 0
        return jsonify({"status": "cache cleared", "cache_type": "redis"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/cache/warm", methods=["POST"])
def warm_cache():
    """Warm Redis cache with common terms"""
    data = request.get_json()
    terms = data.get("terms", [])

    warmed = 0
    failed = 0

    for term in terms:
        try:
            if re.search(r"[\u0600-\u06FF]", term):
                cache_key = f"ar-en-{term}"
                if not cache.get(cache_key):
                    variants = transliterate_with_model(term, "ar", "en")
                    cache.set(cache_key, variants, timeout=CACHE_DEFAULT_TIMEOUT)
                    warmed += 1
            else:
                cache_key = f"en-ar-{term}"
                if not cache.get(cache_key):
                    variants = transliterate_with_model(term, "en", "ar")
                    cache.set(cache_key, variants, timeout=CACHE_DEFAULT_TIMEOUT)
                    warmed += 1
        except Exception as e:
            print(f"Failed to warm cache for '{term}': {e}")
            failed += 1

    return jsonify(
        {"status": "success", "warmed": warmed, "failed": failed, "cache_type": "redis"}
    )


@app.route("/cache/info", methods=["GET"])
def cache_info():
    """Get Redis cache information"""
    try:
        if cache_config.get("CACHE_TYPE") == "RedisCache":
            redis_client = cache.cache._write_client
            info = redis_client.info()

            return jsonify(
                {
                    "cache_type": "Redis",
                    "redis_version": info.get("redis_version"),
                    "used_memory_human": info.get("used_memory_human"),
                    "connected_clients": info.get("connected_clients"),
                    "total_commands_processed": info.get("total_commands_processed"),
                    "keyspace_hits": info.get("keyspace_hits"),
                    "keyspace_misses": info.get("keyspace_misses"),
                    "keys": redis_client.dbsize(),
                }
            )
        else:
            return jsonify({"cache_type": "SimpleCache", "message": "Not using Redis"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 Starting Flask Transliteration API...")
    print(f"📦 Model choice: {MODEL_CHOICE}")
    print("")

    # Load better models
    success = load_models()

    if success:
        print("\n✅ High-quality models loaded!")
        print("   AR→EN: opus-mt-tc-big-ar-en")
        if MODEL_CHOICE == "opus-big":
            print("   EN→AR: opus-mt-tc-big-en-ar (MUCH better than basic!)")
        else:
            print("   EN→AR: marefa-mt-en-ar (Arabic-specialized)")
    else:
        print("\n⚠️  Models failed to load")

    print("\n✅ Server starting on http://localhost:5000")
    print("📝 Endpoints:")
    print("   POST /transliterate")
    print("   POST /transliterate/batch")
    print("   GET  /health")
    print("   GET  /stats")
    print("   GET  /cache/info")
    print("   POST /cache/clear")
    print("   POST /cache/warm")

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
