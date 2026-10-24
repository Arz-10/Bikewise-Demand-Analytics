# Applied Data Science: BikeWise Demand Analytics

A full-stack web application designed for the **Applied Data Science** college subject. The application analyzes the fixed Kaggle **Bike Sharing Demand** dataset and demonstrates key concepts from an Applied Data Science syllabus (Data exploration, descriptive statistics, inferential statistics, regression modeling, anomaly detection, and time-series forecasting).

---

## Technology Stack

### Frontend
- **Framework**: Next.js 15 (App Router)
- **UI Library & Components**: React, Tailwind CSS, shadcn/ui components
- **Visualization & Utilities**: Recharts, Lucide React, date-fns, TypeScript

### Backend
- **Framework**: Python 3.11+, FastAPI, Uvicorn
- **Data Science & Analytics**: pandas, NumPy, SciPy, scikit-learn, statsmodels, openpyxl

---

## Project Structure

```
Applied-Data-Science/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sampleSubmission.csv
│
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── globals.css
│   │   │   ├── layout.tsx
│   │   │   └── page.tsx
│   │   └── lib/
│   │       └── utils.ts
│   ├── package.json
│   └── tsconfig.json
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── data_service.py
│   │   │   ├── statistics_service.py
│   │   │   ├── inference_service.py
│   │   │   ├── regression_service.py
│   │   │   ├── anomaly_service.py
│   │   │   └── timeseries_service.py
│   │   └── routers/
│   │       ├── __init__.py
│   │       ├── dataset.py
│   │       ├── statistics.py
│   │       ├── regression.py
│   │       ├── anomaly.py
│   │       └── timeseries.py
│   ├── requirements.txt
│   └── .gitignore
│
├── .gitignore
└── README.md
```

---

## Prerequisites

- **Node.js**: v18.0.0 or higher (v25+ supported)
- **Python**: v3.11 or v3.12 (Python 3.11.8 installed)
- **npm**: v9.0.0 or higher

---

## Fixed Dataset Placement

The application uses **ONE fixed Kaggle dataset**. Dataset file uploads are strictly disallowed by design.

Place the official Kaggle Bike Sharing Demand CSV files directly into the root `data/` directory:
- `data/train.csv`
- `data/test.csv`
- `data/sampleSubmission.csv`

---

## Setup & Running Instructions

### 1. Backend Setup

Open a terminal and navigate to the `backend/` directory:

```bash
cd backend
```

#### Create & Activate Virtual Environment

- **Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```

- **Windows (Command Prompt)**:
  ```cmd
  python -m venv .venv
  \.venv\Scripts\activate.bat
  ```

- **macOS / Linux**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

#### Install Backend Dependencies

```bash
pip install -r requirements.txt
```

#### Start FastAPI Server

```bash
uvicorn app.main:app --reload --port 8000
```

- **API URL**: `http://localhost:8000`
- **API Health Endpoint**: `http://localhost:8000/api/health`
- **Swagger Documentation**: `http://localhost:8000/docs`

---

### 2. Frontend Setup

Open a separate terminal window and navigate to the `frontend/` directory:

```bash
cd frontend
```

#### Install Frontend Dependencies

```bash
npm install
```

#### Start Next.js Development Server

```bash
npm run dev
```

- **Frontend Application URL**: `http://localhost:3000`

---

## Verification Endpoints

1. **Frontend Build**: Test production build via `cd frontend && npm run build`
2. **Backend Health Check**: Open `http://localhost:8000/api/health` in browser or curl:
   ```json
   {
     "status": "healthy",
     "message": "Backend is running successfully",
     "app": "Applied Data Science - Bike Sharing Demand Analytics"
   }
   ```
