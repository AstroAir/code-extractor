# Sample Web Application

A sample web application built with FastAPI for demonstration and testing purposes.

## Features

- User authentication with JWT tokens
- CRUD operations for users and posts
- Middleware for logging and rate limiting
- Database connection pooling
- Redis caching layer

## Project Structure

```
src/
├── app.py          # Application entry point
├── config.py       # Configuration management
├── models/         # Data models
├── services/       # Business logic
├── api/            # HTTP routes and middleware
├── utils/          # Helper utilities
└── db/             # Database layer
```

## Quick Start

```bash
pip install -e .
uvicorn src.app:app --reload
```

## TODO

- Add WebSocket support
- Implement OAuth2 provider
- Add rate limiting per user
