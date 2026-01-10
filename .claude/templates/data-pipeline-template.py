"""
Data Pipeline Template for BT Backtesting Framework

Template for building robust data pipelines with quality checks,
error handling, and monitoring.

Author: [Your Name]
Created: [Date]
Pipeline: [Description]
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import logging
from pathlib import Path


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """
    Configuration for data pipeline.

    Attributes:
        name: Pipeline name
        source: Data source identifier
        destination: Output location
        frequency: Update frequency ('daily', 'hourly', etc.)
        quality_threshold: Minimum quality score (0-1)
        max_retries: Maximum retry attempts
    """
    name: str = "DataPipeline"
    source: str = "vendor_api"
    destination: str = "data/processed"
    frequency: str = "daily"
    quality_threshold: float = 0.95
    max_retries: int = 3


class DataPipeline:
    """
    Base data pipeline implementation.

    Implements the ETL (Extract-Transform-Load) pattern with:
    - Data extraction from sources
    - Data validation and quality checks
    - Data transformation and enrichment
    - Data loading to storage
    - Error handling and logging
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline with configuration.

        Args:
            config: PipelineConfig object
        """
        self.config = config
        self.name = config.name
        self.metrics = {}

        # Create output directory if needed
        Path(config.destination).mkdir(parents=True, exist_ok=True)

    def extract(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Extract data from source.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Returns:
            Raw data as DataFrame

        Raises:
            Exception: If extraction fails after retries
        """
        logger.info(f"Extracting data from {start_date} to {end_date}")

        for attempt in range(self.config.max_retries):
            try:
                # TODO: Implement data extraction
                # Example: API call, database query, file read
                data = self._fetch_from_source(start_date, end_date)

                logger.info(f"Successfully extracted {len(data)} records")
                self.metrics['records_extracted'] = len(data)

                return data

            except Exception as e:
                logger.error(f"Extraction attempt {attempt + 1} failed: {e}")

                if attempt == self.config.max_retries - 1:
                    logger.error("Max retries reached, extraction failed")
                    raise

                # Exponential backoff
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                # time.sleep(wait_time)

        raise Exception("Extraction failed")

    def _fetch_from_source(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch data from specific source.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Raw data
        """
        # TODO: Implement actual data fetching
        # Examples:
        # - API call: requests.get(url, params={'start': start_date, 'end': end_date})
        # - Database: pd.read_sql(query, connection)
        # - File: pd.read_csv(file_path)

        # Placeholder
        logger.info("Fetching data from source...")
        data = pd.DataFrame()
        return data

    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality.

        Args:
            data: Data to validate

        Returns:
            Dictionary with validation results and quality score
        """
        logger.info("Validating data quality...")

        validation_results = {}

        # 1. Completeness check
        completeness = self._check_completeness(data)
        validation_results['completeness'] = completeness

        # 2. Validity check
        validity = self._check_validity(data)
        validation_results['validity'] = validity

        # 3. Consistency check
        consistency = self._check_consistency(data)
        validation_results['consistency'] = consistency

        # 4. Timeliness check
        timeliness = self._check_timeliness(data)
        validation_results['timeliness'] = timeliness

        # Calculate overall quality score (weighted average)
        quality_score = (
            completeness * 0.30 +
            validity * 0.30 +
            consistency * 0.20 +
            timeliness * 0.20
        )

        validation_results['quality_score'] = quality_score
        self.metrics['quality_score'] = quality_score

        logger.info(f"Quality score: {quality_score:.2%}")

        # Check against threshold
        if quality_score < self.config.quality_threshold:
            logger.warning(
                f"Quality score {quality_score:.2%} below threshold "
                f"{self.config.quality_threshold:.2%}"
            )

        return validation_results

    def _check_completeness(self, data: pd.DataFrame) -> float:
        """
        Check data completeness.

        Args:
            data: Data to check

        Returns:
            Completeness score (0-1)
        """
        if len(data) == 0:
            return 0.0

        # Calculate missing value rate
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isna().sum().sum()
        completeness = 1 - (missing_cells / total_cells)

        logger.info(f"Completeness: {completeness:.2%}")
        return completeness

    def _check_validity(self, data: pd.DataFrame) -> float:
        """
        Check data validity.

        Args:
            data: Data to check

        Returns:
            Validity score (0-1)
        """
        # TODO: Implement domain-specific validation rules

        # Example checks:
        # - Price > 0
        # - Volume >= 0
        # - High >= Low
        # - Date format correct

        # Placeholder: assume all valid
        validity = 1.0

        logger.info(f"Validity: {validity:.2%}")
        return validity

    def _check_consistency(self, data: pd.DataFrame) -> float:
        """
        Check data consistency.

        Args:
            data: Data to check

        Returns:
            Consistency score (0-1)
        """
        # TODO: Implement consistency checks

        # Example checks:
        # - OHLC relationships (High >= Low, etc.)
        # - Cross-field validation
        # - Time series continuity

        consistency = 1.0

        logger.info(f"Consistency: {consistency:.2%}")
        return consistency

    def _check_timeliness(self, data: pd.DataFrame) -> float:
        """
        Check data timeliness.

        Args:
            data: Data to check

        Returns:
            Timeliness score (0-1)
        """
        if len(data) == 0 or 'date' not in data.columns:
            return 0.0

        # Check how recent the data is
        latest_date = pd.to_datetime(data['date']).max()
        age_days = (datetime.now() - latest_date).days

        # Score based on age (fresher = higher score)
        if age_days <= 1:
            score = 1.0
        elif age_days <= 7:
            score = 0.8
        elif age_days <= 30:
            score = 0.6
        else:
            score = 0.4

        logger.info(f"Data age: {age_days} days, Timeliness: {score:.2%}")
        return score

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform and enrich data.

        Args:
            data: Raw data

        Returns:
            Transformed data
        """
        logger.info("Transforming data...")

        transformed = data.copy()

        # TODO: Implement transformations

        # Common transformations:
        # 1. Data type conversions
        # transformed['date'] = pd.to_datetime(transformed['date'])
        # transformed['price'] = transformed['price'].astype(float)

        # 2. Missing value handling
        # transformed = transformed.fillna(method='ffill')

        # 3. Outlier handling
        # transformed = self._handle_outliers(transformed)

        # 4. Derived fields
        # transformed['returns'] = transformed['price'].pct_change()

        # 5. Normalization
        # transformed = self._normalize_prices(transformed)

        self.metrics['records_transformed'] = len(transformed)

        logger.info(f"Transformed {len(transformed)} records")
        return transformed

    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers in data.

        Args:
            data: Data with potential outliers

        Returns:
            Data with outliers handled
        """
        # TODO: Implement outlier detection and handling

        # Example: Z-score method
        # numeric_cols = data.select_dtypes(include=[np.number]).columns
        # z_scores = np.abs((data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std())
        # data[z_scores > 3] = np.nan  # Mark outliers as NaN

        return data

    def load(self, data: pd.DataFrame, date: datetime) -> None:
        """
        Load data to storage.

        Args:
            data: Processed data
            date: Reference date for partitioning
        """
        logger.info("Loading data to storage...")

        # TODO: Implement data loading

        # Create partition path
        partition_path = Path(self.config.destination) / f"year={date.year}" / f"month={date.month:02d}"
        partition_path.mkdir(parents=True, exist_ok=True)

        # Save to file
        output_file = partition_path / f"data_{date.strftime('%Y%m%d')}.parquet"

        try:
            data.to_parquet(output_file, compression='snappy')
            logger.info(f"Saved data to {output_file}")

            self.metrics['records_loaded'] = len(data)
            self.metrics['output_file'] = str(output_file)

        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            raise

    def run(self, start_date: datetime, end_date: datetime) -> Dict:
        """
        Run the complete pipeline.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary with pipeline metrics
        """
        pipeline_start = datetime.now()
        logger.info(f"Starting pipeline: {self.name}")

        try:
            # Extract
            raw_data = self.extract(start_date, end_date)

            # Validate
            validation_results = self.validate(raw_data)

            # Check quality threshold
            if validation_results['quality_score'] < self.config.quality_threshold:
                logger.error("Data quality below threshold, pipeline stopped")
                self.metrics['status'] = 'failed'
                self.metrics['reason'] = 'quality_threshold_not_met'
                return self.metrics

            # Transform
            transformed_data = self.transform(raw_data)

            # Load
            self.load(transformed_data, end_date)

            # Success
            pipeline_end = datetime.now()
            execution_time = (pipeline_end - pipeline_start).total_seconds()

            self.metrics['status'] = 'success'
            self.metrics['execution_time_seconds'] = execution_time
            self.metrics['start_time'] = pipeline_start.isoformat()
            self.metrics['end_time'] = pipeline_end.isoformat()

            logger.info(f"Pipeline completed successfully in {execution_time:.2f}s")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.metrics['status'] = 'failed'
            self.metrics['error'] = str(e)
            raise

        return self.metrics


def run_pipeline(config: PipelineConfig, date: Optional[datetime] = None) -> Dict:
    """
    Convenience function to run pipeline.

    Args:
        config: Pipeline configuration
        date: Target date (defaults to today)

    Returns:
        Pipeline metrics
    """
    if date is None:
        date = datetime.now()

    # Run for single day
    start_date = date
    end_date = date

    pipeline = DataPipeline(config)
    metrics = pipeline.run(start_date, end_date)

    return metrics


if __name__ == "__main__":
    """
    Example usage and testing.
    """
    # Create configuration
    config = PipelineConfig(
        name="Example Data Pipeline",
        source="example_api",
        destination="data/processed",
        frequency="daily",
        quality_threshold=0.95
    )

    # Run pipeline
    try:
        metrics = run_pipeline(config)
        print(f"Pipeline metrics: {metrics}")
    except Exception as e:
        print(f"Pipeline failed: {e}")
