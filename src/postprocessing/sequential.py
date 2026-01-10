"""Sequential post-processing for page classifications."""
import numpy as np
from typing import List, Dict, Tuple
import pandas as pd


class SequentialPostProcessor:
    """Post-process predictions using sequential page order logic."""

    def __init__(
        self,
        min_article_length: int = 2,
        isolated_correction: bool = True,
        blank_tolerance: int = 1,
        label_names: List[str] = None
    ):
        """Initialize post-processor.

        Args:
            min_article_length: Minimum pages for valid article
            isolated_correction: Correct isolated single-page articles
            blank_tolerance: Allow N blank pages within article sequence
            label_names: List of label names
        """
        self.min_article_length = min_article_length
        self.isolated_correction = isolated_correction
        self.blank_tolerance = blank_tolerance

        if label_names is None:
            self.label_names = [
                'article_start', 'article_continuation', 'article_end',
                'illustrated_plate', 'plate_caption', 'blank_page', 'other'
            ]
        else:
            self.label_names = label_names

        # Get indices for each label
        self.idx_article_start = self.label_names.index('article_start')
        self.idx_article_cont = self.label_names.index('article_continuation')
        self.idx_article_end = self.label_names.index('article_end')
        self.idx_blank = self.label_names.index('blank_page')

    def process_sequence(
        self,
        predictions: np.ndarray,
        filenames: List[str] = None,
        page_nums: List[int] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Process a sequence of predictions.

        Args:
            predictions: Binary predictions [num_pages, num_labels]
            filenames: List of filenames (optional, for debugging)
            page_nums: List of page numbers (optional, for debugging)

        Returns:
            Tuple of (corrected_predictions, statistics)
        """
        corrected = predictions.copy()
        stats = {
            'total_pages': len(predictions),
            'corrections_made': 0,
            'isolated_articles_fixed': 0,
            'boundary_corrections': 0,
            'continuation_insertions': 0
        }

        # 1. Fix isolated single-page articles
        if self.isolated_correction:
            corrected, isolated_fixes = self._fix_isolated_articles(corrected)
            stats['isolated_articles_fixed'] = isolated_fixes
            stats['corrections_made'] += isolated_fixes

        # 2. Ensure article boundaries are consistent
        corrected, boundary_fixes = self._fix_article_boundaries(corrected)
        stats['boundary_corrections'] = boundary_fixes
        stats['corrections_made'] += boundary_fixes

        # 3. Fill gaps in article sequences
        corrected, cont_fixes = self._fill_article_gaps(corrected)
        stats['continuation_insertions'] = cont_fixes
        stats['corrections_made'] += cont_fixes

        # 4. Handle blank pages within articles
        corrected = self._handle_blank_pages(corrected)

        return corrected, stats

    def _fix_isolated_articles(
        self,
        predictions: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """Fix isolated single-page articles that are suspicious.

        Args:
            predictions: Binary predictions

        Returns:
            Tuple of (corrected predictions, number of fixes)
        """
        corrected = predictions.copy()
        num_fixes = 0

        for i in range(len(predictions)):
            # Check if this page has start AND end labels
            has_start = corrected[i, self.idx_article_start] == 1
            has_end = corrected[i, self.idx_article_end] == 1

            if has_start and has_end:
                # Check if previous/next pages are part of articles
                prev_in_article = (
                    i > 0 and
                    corrected[i - 1, self.idx_article_cont] == 1
                )
                next_in_article = (
                    i < len(predictions) - 1 and
                    corrected[i + 1, self.idx_article_cont] == 1
                )

                # If surrounded by article pages, this is suspicious
                if prev_in_article or next_in_article:
                    # Change to continuation only
                    corrected[i, self.idx_article_start] = 0
                    corrected[i, self.idx_article_end] = 0
                    corrected[i, self.idx_article_cont] = 1
                    num_fixes += 1

        return corrected, num_fixes

    def _fix_article_boundaries(
        self,
        predictions: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """Ensure article start/end boundaries are consistent.

        Args:
            predictions: Binary predictions

        Returns:
            Tuple of (corrected predictions, number of fixes)
        """
        corrected = predictions.copy()
        num_fixes = 0

        # Track if we're currently in an article
        in_article = False

        for i in range(len(predictions)):
            has_start = corrected[i, self.idx_article_start] == 1
            has_cont = corrected[i, self.idx_article_cont] == 1
            has_end = corrected[i, self.idx_article_end] == 1

            # Article start without being in article
            if has_start and not in_article:
                in_article = True

            # Continuation without article start
            elif has_cont and not in_article and not has_start:
                # Add start label
                corrected[i, self.idx_article_start] = 1
                in_article = True
                num_fixes += 1

            # Article end
            if has_end:
                in_article = False

            # Still in article but no continuation or end
            elif in_article and not has_cont and not has_end and not has_start:
                # Add continuation label
                corrected[i, self.idx_article_cont] = 1
                num_fixes += 1

        return corrected, num_fixes

    def _fill_article_gaps(
        self,
        predictions: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """Fill gaps in article sequences with continuation labels.

        Args:
            predictions: Binary predictions

        Returns:
            Tuple of (corrected predictions, number of fixes)
        """
        corrected = predictions.copy()
        num_fixes = 0

        # Find article segments
        i = 0
        while i < len(predictions):
            # Find article start
            if corrected[i, self.idx_article_start] == 1:
                start_idx = i
                # Find article end
                end_idx = None
                for j in range(i + 1, len(predictions)):
                    if corrected[j, self.idx_article_end] == 1:
                        end_idx = j
                        break

                if end_idx is not None:
                    # Fill any gaps between start and end
                    for k in range(start_idx + 1, end_idx):
                        if corrected[k, self.idx_article_cont] == 0:
                            # Check if it's a blank page (might be intentional gap)
                            if corrected[k, self.idx_blank] == 0:
                                corrected[k, self.idx_article_cont] = 1
                                num_fixes += 1

                    i = end_idx + 1
                else:
                    i += 1
            else:
                i += 1

        return corrected, num_fixes

    def _handle_blank_pages(
        self,
        predictions: np.ndarray
    ) -> np.ndarray:
        """Handle blank pages within article sequences.

        Args:
            predictions: Binary predictions

        Returns:
            Corrected predictions
        """
        corrected = predictions.copy()

        # If blank page appears within article sequence,
        # it might be intentional (e.g., chapter break)
        # Keep blank label but remove article labels
        for i in range(len(predictions)):
            if corrected[i, self.idx_blank] == 1:
                # Check if surrounded by article pages
                prev_article = (
                    i > 0 and
                    (corrected[i - 1, self.idx_article_start] == 1 or
                     corrected[i - 1, self.idx_article_cont] == 1)
                )
                next_article = (
                    i < len(predictions) - 1 and
                    (corrected[i + 1, self.idx_article_cont] == 1 or
                     corrected[i + 1, self.idx_article_end] == 1)
                )

                if prev_article and next_article:
                    # Keep blank label, remove article labels
                    corrected[i, self.idx_article_start] = 0
                    corrected[i, self.idx_article_cont] = 0
                    corrected[i, self.idx_article_end] = 0

        return corrected

    def process_by_publication(
        self,
        predictions: np.ndarray,
        publication_ids: List[str],
        filenames: List[str] = None,
        page_nums: List[int] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Process predictions grouped by publication.

        Args:
            predictions: Binary predictions
            publication_ids: List of publication IDs for each page
            filenames: List of filenames
            page_nums: List of page numbers

        Returns:
            Tuple of (corrected predictions, statistics)
        """
        corrected = np.zeros_like(predictions)
        all_stats = []

        # Group by publication
        unique_pubs = sorted(set(publication_ids))

        for pub_id in unique_pubs:
            # Get indices for this publication
            indices = [i for i, pid in enumerate(publication_ids) if pid == pub_id]

            # Extract predictions for this publication
            pub_predictions = predictions[indices]

            # Process this sequence
            pub_corrected, pub_stats = self.process_sequence(
                pub_predictions,
                filenames=[filenames[i] for i in indices] if filenames else None,
                page_nums=[page_nums[i] for i in indices] if page_nums else None
            )

            # Put back corrected predictions
            for i, idx in enumerate(indices):
                corrected[idx] = pub_corrected[i]

            all_stats.append(pub_stats)

        # Aggregate statistics
        combined_stats = {
            'total_publications': len(unique_pubs),
            'total_pages': sum(s['total_pages'] for s in all_stats),
            'corrections_made': sum(s['corrections_made'] for s in all_stats),
            'isolated_articles_fixed': sum(s['isolated_articles_fixed'] for s in all_stats),
            'boundary_corrections': sum(s['boundary_corrections'] for s in all_stats),
            'continuation_insertions': sum(s['continuation_insertions'] for s in all_stats)
        }

        return corrected, combined_stats

    def predictions_to_dataframe(
        self,
        predictions: np.ndarray,
        filenames: List[str],
        page_nums: List[int] = None,
        publication_ids: List[str] = None
    ) -> pd.DataFrame:
        """Convert predictions to readable DataFrame.

        Args:
            predictions: Binary predictions [num_pages, num_labels]
            filenames: List of filenames
            page_nums: List of page numbers
            publication_ids: List of publication IDs

        Returns:
            DataFrame with predictions
        """
        data = {'filename': filenames}

        if page_nums is not None:
            data['page_num'] = page_nums

        if publication_ids is not None:
            data['publication_id'] = publication_ids

        # Add predictions for each label
        for i, label_name in enumerate(self.label_names):
            data[label_name] = predictions[:, i]

        return pd.DataFrame(data)
