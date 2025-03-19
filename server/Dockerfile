# setting Base Image
FROM python:3.10-slim

# Install poetry
RUN pip install poetry

# set the Working Directory
WORKDIR /app

# Copy project files
COPY pyproject.toml poetry.lock ./

# Configure poetry and install dependencies
RUN poetry lock \
    && poetry config virtualenvs.create false \  
    && poetry install --no-interaction --no-ansi --no-root
 
# copy the Application Code
COPY . . 

# expose the Port for server/backend
EXPOSE 8000