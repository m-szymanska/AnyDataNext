# AnyDataNext Frontend

This directory contains the Next.js application providing the web interface for AnyDataNext.

## Prerequisites

- Node.js 18+ (with `npm`)
- The backend API running locally on port `8000` (use `./backend-start.sh` from the project root)

## Quick Start

1. Install dependencies (only needed once):

   ```bash
   npm install
   ```

2. Create a `.env.local` file with the backend URL (optional if using default):

   ```bash
   echo "NEXT_PUBLIC_BACKEND_URL=http://localhost:8000" > .env.local
   ```

3. Start the development server:

   ```bash
   npm run dev
   ```

   Alternatively you can run `../frontend-start.sh` from the project root which
   checks that the backend is running before launching the dev server.

4. Open <http://localhost:3000> in your browser.

## Debugging Tips

- The console running `npm run dev` shows build errors and warnings.
- Check the browser console for WebSocket messages and API errors.
- If the UI cannot connect to the backend, verify the `NEXT_PUBLIC_BACKEND_URL`
  value in `.env.local`.

