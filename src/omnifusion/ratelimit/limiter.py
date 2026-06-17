import asyncio
import time
from typing import Dict


class TokenBucket:
    def __init__(self, rate: float, capacity: float):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def consume(self, tokens: float = 1.0) -> bool:
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            self.last_refill = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    async def wait_consume(self, tokens: float = 1.0):
        while True:
            if await self.consume(tokens):
                return
            await asyncio.sleep(0.05)


class RateLimiter:
    def __init__(self):
        # Global concurrency limit
        self.global_semaphore = asyncio.Semaphore(50)
        # Per provider semaphores (concurrency)
        self.provider_semaphores: Dict[str, asyncio.Semaphore] = {}
        # Per provider token buckets (rate)
        self.provider_buckets: Dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()

    async def get_provider_limiter(self, provider_id: str):
        async with self._lock:
            if provider_id not in self.provider_semaphores:
                # Limit each provider to 10 concurrent requests by default
                self.provider_semaphores[provider_id] = asyncio.Semaphore(10)
            if provider_id not in self.provider_buckets:
                # 5 requests per second, burst 10 by default
                self.provider_buckets[provider_id] = TokenBucket(
                    rate=5.0, capacity=10.0
                )
            return self.provider_semaphores[provider_id], self.provider_buckets[
                provider_id
            ]

    async def acquire(self, provider_id: str):
        sem, bucket = await self.get_provider_limiter(provider_id)
        # 1. Enforce rate limit (token bucket)
        await bucket.wait_consume(1.0)
        # 2. Enforce provider concurrency first (prevents HoL blocking on global sem)
        await sem.acquire()
        try:
            # 3. Enforce global concurrency
            await self.global_semaphore.acquire()
        except BaseException:
            sem.release()
            raise

    def release(self, provider_id: str):
        if provider_id in self.provider_semaphores:
            self.provider_semaphores[provider_id].release()
        self.global_semaphore.release()


rate_limiter = RateLimiter()
