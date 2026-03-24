# Frontend development Dockerfile
# Supports: build, dev server, prettier formatting, integration tests

FROM node:22-alpine AS base
WORKDIR /app

# Install dependencies
FROM base AS deps
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci || npm install

# Development with hot reload
FROM deps AS dev
COPY frontend/ .
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

# Format code with prettier (outputs to mounted volume)
FROM deps AS format
COPY frontend/ .
CMD ["npm", "run", "format"]

# Type check
FROM deps AS typecheck
COPY frontend/ .
CMD ["npx", "tsc", "-b"]

# Integration tests (requires SERVER_URL env var)
FROM deps AS integration-test
COPY frontend/ .
CMD ["npm", "run", "test:integration"]

# Build production (for standalone build)
FROM deps AS builder
COPY frontend/ .
RUN npm run build

# Production server (nginx)
FROM nginx:alpine AS prod
COPY --from=builder /app/dist /usr/share/nginx/html
COPY dockerfile/frontend/nginx.frontend.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
