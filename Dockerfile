FROM python:3.10-slim  
  
WORKDIR /app  
  
RUN pip install --no-cache-dir \  
    pandas \  
    numpy \  
    numba \  
    joblib \  
    mlflow-skinny \  
    fastapi \  
    uvicorn  
  
COPY multifactor_portfolio /app/multifactor_portfolio  
  
EXPOSE 8000  
  
CMD ["uvicorn", "multifactor_portfolio.scoring.serve:app", "--host", "0.0.0.0", "--port", "8000"]  
