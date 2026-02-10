FROM --platform=$TARGETPLATFORM node:20-bookworm-slim

WORKDIR /app

# System libs often required by the libwebrtc build used by `wrtc` on Debian/Ubuntu.
# Keep this explicit to avoid "works locally, fails on RunPod" surprises.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libnss3 \
    libx11-xcb1 \
    libasound2 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libglib2.0-0 \
    libgtk-3-0 \
    libnspr4 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libcups2 \
    libdrm2 \
    libgbm1 \
    libx11-6 \
    libxcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libxrender1 \
    libxss1 \
    libxtst6 \
  && rm -rf /var/lib/apt/lists/*

COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY server.js ./

ENV NODE_ENV=production
ENV PORT=3000

EXPOSE 3000

CMD ["node", "server.js"]


