# Project Setup Instructions

Follow these steps to set up the project environment:

## Commands to Set Up the Environment

1. **Create the Virtual Environment**:
   ```bash
   python3 -m venv bml
   ```

2. **Initialize (Activate) the Virtual Environment**:

    ```bash
    source bml/bin/activate
    ```

3. **Install Required Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure neptune**:
   ```bash
   touch .env
   echo -e "NEPTUNE_PROJECT=your_workspace/your_project" >> .env
   echo -e "NEPTUNE_API_TOKEN=your_neptune_api_token" >> .env
   ```
