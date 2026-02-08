#!/usr/bin/env python3
"""Seed the database with sample data for development.

This script creates initial users, posts, and other test data
to facilitate local development and testing.
"""

from __future__ import annotations

import asyncio
from typing import Any


async def create_sample_users() -> list[dict[str, Any]]:
    """Create a set of sample user accounts.

    Returns:
        List of created user data dictionaries.
    """
    users = [
        {"username": "admin", "email": "admin@example.com", "role": "admin"},
        {"username": "alice", "email": "alice@example.com", "role": "moderator"},
        {"username": "bob", "email": "bob@example.com", "role": "user"},
        {"username": "charlie", "email": "charlie@example.com", "role": "user"},
        {"username": "guest", "email": "guest@example.com", "role": "guest"},
    ]
    print(f"Created {len(users)} sample users")
    return users


async def create_sample_posts(users: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create sample blog posts for each user.

    Args:
        users: List of user data to create posts for.

    Returns:
        List of created post data dictionaries.
    """
    posts = []
    for user in users:
        if user["role"] in ("admin", "moderator", "user"):
            post = {
                "title": f"Hello from {user['username']}",
                "content": f"This is a sample post by {user['username']}.",
                "author": user["username"],
                "tags": ["sample", user["role"]],
            }
            posts.append(post)
    print(f"Created {len(posts)} sample posts")
    return posts


async def seed_database() -> None:
    """Run the complete seeding process."""
    print("Starting database seeding...")
    users = await create_sample_users()
    posts = await create_sample_posts(users)
    print(f"Seeding complete: {len(users)} users, {len(posts)} posts")


if __name__ == "__main__":
    asyncio.run(seed_database())
