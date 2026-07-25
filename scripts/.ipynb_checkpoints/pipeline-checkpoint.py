import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path as p
from datetime import datetime
root_folder = p.cwd().parent

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("DataPipeline")

# step 1 ingestion
class DataIngestor:
    """
    Handles loading data from multiple file formats into a single Pandas dataframe
    regardless of whether the source file is .csv,.xslx,.json 
    """
    SUPPORTED_EXTENSIONS = {
        ".csv",".xlsx",".xls",".json"
    }
    def load(self,filepath : str) ->pd.DataFrame:

        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f" File not Found {filepath}")
        
        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported File Type '{ext}.'"
                f"Only Supports {self.SUPPORTED_EXTENSIONS}"
            )
        logger.info("Ingesting file : {path.name} type : {.ext}")

        try:
            if ext == ".csv":
                df = pd.read_csv(path)
            elif ext == (".xlsx",".xls"):
                df = pd.read_excel(path)
            elif ext == ".json":
                with open(path,"r",encoding = 'utf-8') as f:
                    raw = json.load(f)
                    df = pd.json_normalize(raw)
        except Exception as e:
            logger.error(f"Failed to load {path.name} : {e}")
            raise 
        logger.info(f"Loaded {path.name}: {df.shape[0]} rows, {df.shape[1]} cols")
        return df
    def load_multiple(self,filepaths:list)->pd.DataFrame:
        """
        loads and concatenates multiple files of possible mixed formats into one DataFrame.
        """
        dfs = [self.load(fp) for fp in filepaths]
        combined = pd.concat(dfs,ignore_index = True,sort = False)
        logger.info(f"Combined {len(filepaths)} files into one dataset:" f"{combined.shape[0]} rows , {combined.shape[1]} cols")
        return combined
class ValidationReport:
    """
    Tracks every cleaning action taken,plus before ,after data query stats.
    """
    def __init__(self):
        self.action = []
        self.start_shape = None
        self.end_shape = None
    
    def log_action(self,message:str):
        self.action.append(message)
        logger.info(message)
    def generate(self,df_before:DataFrame,df_after:DataFrame)->str:
        lines = []
        lines.append("="*70)
        lines.append("DATA CLEANING VALIDATION REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("="*70)
        lines.append(f"\n Shape Before Cleaning : {df_before.shape}")
        lines.append(f"\n Shape After Cleaning : {df_after.shape}")
        lines.append(f"\n Rows Removed : {df_before.shape[0]} - {df_after.shape[0]}")
        lines.append(f"\n Columns Changed : {df_before.shape[1]} -> {df_after.shape[1]}")
        lines.append("\n------ Actions Taken ------")

        for i,action in enumerate(self.action,1):
            lines.append(f"{i}. {action}")
        lines.append("\n--- Missing Values (after cleaning) ---")
        missing = df_after.isnull().sum()
        missing = missing[missing>0]
        if missing.empty:
            lines.append("None Remaining.")
        else:    
            for col, count in missing.items():
                pct = round(count / len(df_after) * 100, 2)
                lines.append(f"{col}: {count} missing ({pct}%)")