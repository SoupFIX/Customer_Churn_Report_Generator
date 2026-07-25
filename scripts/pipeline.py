import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from datetime import datetime
root_folder = Path.cwd().parent

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
        logger.info(f"Ingesting file : {path.name} type : {ext}")

        try:
            if ext == ".csv":
                df = pd.read_csv(path)

            elif ext in (".xlsx",".xls"):
                df = pd.read_excel(path)

            elif ext == ".json":
                with open(path,"r",encoding = 'utf-8') as f:
                    raw = json.load(f)
                    df = pd.json_normalize(raw)

            else:
                raise ValueError(f"No loader implemented for extension: {ext}")
            logger.info(f"Loaded {path.name}: {df.shape[0]} rows, {df.shape[1]} cols")
        except Exception as e:
            logger.error(f"Failed to load {path.name} : {e}")
            raise 
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

        lines.append("\n--- Column Dtypes (after cleaning) ---")
        for col, dtype in df_after.dtypes.items():
            lines.append(f"{col}: {dtype}")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)



class DataCleaner:
    """
    Configurable, step-by-step cleaning pipeline. Each method does ONE
    job and returns the modified DataFrame, so steps can be reordered,
    skipped, or extended easily.

    HOW TO USE:
        cleaner = DataCleaner()
        clean_df = cleaner.run_pipeline(raw_df)
    """

    def __init__(self, outlier_method: str = "iqr", date_columns: list = None):
        """
        outlier_method : 'iqr' or 'zscore' — method used to flag/cap outliers
        date_columns   : list of column names to force-parse as dates
                    (if None, the pipeline will try to auto-detect them)
        """
        self.outlier_method = outlier_method
        self.date_columns = date_columns or []
        self.report = ValidationReport()
        
    def clean_column_names(self,df:pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column name: lowercase,strips whitespace,replaces space/special chars with underscore.
        """
        original_cols = df.columns.tolist()
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(r"[^\w\s]", "", regex=True)
            .str.replace(r"\s+", "_", regex=True)
        )

        if df.columns.duplicated().any():
            duplicate = df.columns[df.columns.duplicated()].unique().tolist()
            df = df.loc[:, ~df.columns.duplicated()]  # keep only the first occurrence
            self.report.log_action(
                f"Found and removed duplicate columns after name standardization: {duplicate}"
            )


        if list(df.columns) != original_cols:
            self.report.log_action("Standardized column names (lowercase,underscores)")
        return df

    #--------------duplicates-----------------
    def remove_duplicates(self,df:pd.DataFrame)->pd.DataFrame:
        """
        Removes fully duplicate rows.
        WHY: Duplicate transactions/records inflate counts (e.g. inflate
        'frequency' in RFM features) and bias model training.
        """
        before = len(df)
        df = df.drop_duplicates()
        removed = before-len(df)
        if removed > 0:
            self.report.log_action(f"Removed {removed} duplicate rows")
        return df

    def clean_text_columns(self,df : pd.DataFrame) -> pd.DataFrame:
            """
            For all text/object columns: strips whitespace, normalizes
            casing inconsistencies, and replaces common "empty" placeholders
            (like '', 'NA', 'null', 'None', '-', 'n/a') with proper NaN so
            pandas' missing-value tools can catch them.
            """
            text_cols = df.select_dtypes(include=["object","string"]).columns
            placeholders = ["","na","n/a","null","none","-","--","nan"]
            for col in text_cols:
                df[col] = df[col].where(df[col].isna(),df[col].astype(str).str.strip())
                df[col] = df[col].apply(
                    lambda x:np.nan if isinstance(x,str) and x.lower() in placeholders else x
                )
            if len(text_cols)>0:
                self.report.log_action(
                    f" CLeaned {len(text_cols)} text column(s) : trimmed whitespace,"
                    f" normalized placeholder values to NaN "
                )
            return df

    def parse_dates(self,df:pd.DataFrame) -> pd.DataFrame:
        """
        Auto-detects and parses date-like columns into proper datetime
        dtype. If self.date_columns is given, only those are forced.
        """
        candidate_cols = self.date_columns or [
            col for col in df.columns
            if "date" in col.lower() or "time" in col.lower()
        ]
        parsed = []
        for col in candidate_cols:
            if col not in df.columns:
                continue
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                parsed.append(col)
            except Exception as e:
                logger.warning(f"Could not parse '{col}' as date: {e}")
        if parsed:
            self.report.log_action(f"Parsed date columns: {parsed}")
        return df

    def fix_dtypes(self,df:pd.DataFrame) -> pd.DataFrame:
        """
        Attempts to downcast/convert object columns that are actually
        numeric (e.g. '1,200' or '42' stored as strings) into proper
        numeric dtypes.
        """
        converted = []
        # finds the texts/object columns
        for col in df.select_dtypes(include = ['object','string']).columns:
            # Relaces any commas in those columns
            cleaned_series = df[col].astype(str).str.replace(",","",regex = False)
            # finds numeric columns from above
            numeric_attempt = pd.to_numeric(cleaned_series,errors="coerce")
            # finds if any column has zero values in it
            non_null_original = df[col].notna().sum()
            if non_null_original > 0:
                success_rate = numeric_attempt.notna().sum() / non_null_original
                if success_rate > 0.9:
                    df[col] = numeric_attempt
                    converted.append(col)
        if converted:
            self.report.log_action(f"Converted to numeric dtype : {converted}")
        return df

    def handle_missing_values(self,df:pd.DataFrame,stratergy: str = "auto",threshold:float=0.5)->pd.DataFrame:
        """
        strategy = 'auto':
        - Drops columns missing more than `threshold` (default 50%) of values
                (WHY: a column that's mostly empty carries little signal and
                often does more harm than good if imputed)

        """
        missing_ratio = df.isnull().mean()
        cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            self.report.log_action(
                f"Dropped columns with >{int(threshold*100)}% missing: {cols_to_drop}"
            )

        # Impute remaining missing values
        numeric_cols = df.select_dtypes(include=np.number).columns
        categorical_cols = df.select_dtypes(include=["object", "string"]).columns 
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                self.report.log_action(
                    f"Filled missing values in '{col}' with median ({median_val:.2f})"
                )

        for col in categorical_cols:
            if df[col].isnull().any():
                mode_series = df[col].mode()
                fill_val = mode_series.iloc[0] if not mode_series.empty else "unknown"
                df[col] = df[col].fillna(fill_val)
                self.report.log_action(
                    f"Filled missing values in '{col}' with mode ('{fill_val}')"
                )
        return df
# ------- --------- Outlier handling ---------------------
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Caps (winsorizes) outliers in numeric columns rather than
        deleting rows — deleting rows loses information from OTHER
        columns in that row, capping preserves the row while limiting
        the outlier's influence.

        IQR method (default): values beyond 1.5x the interquartile range
        get capped at the boundary.
        Z-score method: values beyond 3 standard deviations get capped.
        """
        numeric_cols = df.select_dtypes(include=np.number).columns
        capped_cols = []

        for col in numeric_cols:
            if self.outlier_method == "iqr":
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            else:  # zscore
                mean, std = df[col].mean(), df[col].std()
                lower, upper = mean - 3 * std, mean + 3 * std

            n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            if n_outliers > 0:
                df[col] = df[col].clip(lower=lower, upper=upper)
                capped_cols.append((col, int(n_outliers)))

        if capped_cols:
            details = ", ".join([f"{c}({n})" for c, n in capped_cols])
            self.report.log_action(
                f"Capped outliers ({self.outlier_method} method) in: {details}"
            )
        return df

    def run_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs all cleaning steps in a sensible order:
        1. Clean column names        (so all later steps reference correct names)
        2. Remove exact duplicates   (before wasting effort cleaning junk rows)
        3. Clean text columns        (normalize placeholders to real NaN)
        4. Parse dates
        5. Fix numeric dtypes
        6. Handle missing values     (after text cleanup, so placeholders count as missing)
        7. Handle outliers           (only meaningful once dtypes are correct)
        """
        df_before = df.copy()
        logger.info("Starting cleaning pipeline...")

        df = self.clean_column_names(df)
        df = self.remove_duplicates(df)
        df = self.clean_text_columns(df)
        df = self.parse_dates(df)
        df = self.fix_dtypes(df)
        df = self.handle_missing_values(df)
        df = self.handle_outliers(df)

        logger.info("Cleaning pipeline complete.")
        report_text = self.report.generate(df_before, df)
        print("\n" + report_text)
        return df

if __name__ == "__main__":
    """
    EXAMPLE USAGE:

    Single file:
        ingestor = DataIngestor()
        raw_df = ingestor.load("data/customers.csv")

    Multiple mixed-format files combined into one dataset:
        raw_df = ingestor.load_multiple([
            "data/january.csv",
            "data/february.xlsx",
            "data/march.json",
        ])

    Then clean:
        cleaner = DataCleaner(outlier_method="iqr", date_columns=["order_date"])
        clean_df = cleaner.run_pipeline(raw_df)
        clean_df.to_csv("data/cleaned_output.csv", index=False)
    """

    import argparse

    parser = argparse.ArgumentParser(
        description="Automated data ingestion & cleaning pipeline"
    )
    parser.add_argument(
        "files", nargs="+", help="Path(s) to input file(s): .csv, .xlsx, .json"
    )
    parser.add_argument(
        "--output", default=None, help="Path to save cleaned CSV"
    )
    parser.add_argument(
        "--date-columns",
        nargs="*",
        default=None,
        help="Explicit column names to parse as dates (optional)",
    )
    parser.add_argument(
        "--outlier-method",
        choices=["iqr", "zscore"],
        default="iqr",
        help="Outlier handling method (default: iqr)",
    )
    args = parser.parse_args()
    if args.output is None:
        input_ext = Path(args.files[0]).suffix  # e.g. ".xlsx", ".json", ".csv"
        args.output = f"cleaned{input_ext}"
    else:
        args.output = args.output  # user explicitly overrode it, respect their choice
    ingestor = DataIngestor()
    raw_df = (
        ingestor.load(args.files[0])
        if len(args.files) == 1
        else ingestor.load_multiple(args.files)
    )

    cleaner = DataCleaner(
        outlier_method=args.outlier_method, date_columns=args.date_columns
    )
    clean_df = cleaner.run_pipeline(raw_df)
    output_path = Path(args.output)

    if output_path.suffix == ".csv":
        clean_df.to_csv(output_path, index=False)
    elif output_path.suffix in (".xlsx", ".xls"):
        clean_df.to_excel(output_path, index=False)
    elif output_path.suffix == ".json":
        clean_df.to_json(output_path, orient="records")
    else:
        clean_df.to_csv(output_path, index=False)
    logger.info(f"Cleaned data saved to: {args.output}")