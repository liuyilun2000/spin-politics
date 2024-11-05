import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Union, Optional



class PoliticalSpeechAnalyzer:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the analyzer with a DataFrame containing political speech data.
        
        Args:
            df: DataFrame with columns including 'party', 'country', 'sentence', 'embedding'
        """
        self.df = df
        self.party_encoder = LabelEncoder()
        self.n_layers = df['embedding'].iloc[0].shape[0]
        self.filtered_df = None  # Will store filtered DataFrame
        
    def analyze_dataset_composition(self) -> Dict:
        """
        Analyze basic composition of the dataset.
        
        Returns:
            Dictionary containing various statistics about the dataset
        """
        # Basic counts
        stats = {
            'total_speeches': len(self.df),
            'unique_parties': len(self.df['party'].unique()),
            'countries': self.df['country'].value_counts().to_dict(),
            'parties_by_country': {
                country: len(self.df[self.df['country'] == country]['party'].unique())
                for country in self.df['country'].unique()
            }
        }
        
        # Party statistics
        party_stats = self.df['party'].value_counts()
        stats['speeches_per_party'] = party_stats.to_dict()
        stats['party_percentages'] = (party_stats / len(self.df) * 100).round(2).to_dict()
        
        # Additional demographics if available
        if 'gender' in self.df.columns:
            stats['gender_distribution'] = self.df['gender'].value_counts().to_dict()
        
        if 'ruling' in self.df.columns:
            stats['ruling_party_distribution'] = self.df['ruling'].value_counts().to_dict()
            
        # Speech length statistics
        stats['speech_length'] = {
            'mean': self.df['sentence'].str.len().mean(),
            'median': self.df['sentence'].str.len().median(),
            'min': self.df['sentence'].str.len().min(),
            'max': self.df['sentence'].str.len().max()
        }
        
        # Time period if dates are available
        if 'date' in self.df.columns:
            stats['time_period'] = {
                'start': self.df['date'].min(),
                'end': self.df['date'].max()
            }
        
        return stats

    def print_dataset_statistics(self):
        """
        Print a formatted summary of the dataset statistics with aligned columns.
        """
        stats = self.analyze_dataset_composition()
        
        print("\nDataset Statistics:")
        print("="* 80)
        print(f"Total speeches: {stats['total_speeches']:,}")
        print(f"Unique parties: {stats['unique_parties']}")
        
        print("\nSpeeches by Country:")
        print("="* 80)
        # Format: country(30) | count(10) | percentage(10) | parties(10)
        print(f"{'Country':<30} {'Speeches':>10} {'Percent':>10} {'Parties':>10}")
        print("-" * 80)
        for country, count in sorted(stats['countries'].items(), key=lambda x: x[1], reverse=True):
            print(f"{country:<30} {count:>10,d} {count/stats['total_speeches']*100:>9.1f}% {stats['parties_by_country'][country]:>10d}")
        
        print("\nSpeeches by Party:")
        print("="* 80)
        # Format: party(40) | count(10) | percentage(10)
        print(f"{'Party':<40} {'Speeches':>10} {'Percent':>10}")
        print("-" * 80)
        for party, count in sorted(stats['speeches_per_party'].items(), key=lambda x: x[1], reverse=True):
            print(f"{party:<40} {count:>10,d} {stats['party_percentages'][party]:>9.1f}%")
        
        if 'speech_length' in stats:
            print("\nSpeech Length Statistics:")
            print("="* 80)
            print(f"{'Metric':<20} {'Value':>10}")
            print("-" * 80)
            print(f"{'Mean length:':<20} {stats['speech_length']['mean']:>10.1f}")
            print(f"{'Median length:':<20} {stats['speech_length']['median']:>10.1f}")
            print(f"{'Minimum length:':<20} {stats['speech_length']['min']:>10d}")
            print(f"{'Maximum length:':<20} {stats['speech_length']['max']:>10d}")
        
        if 'time_period' in stats:
            print("\nTime Period:")
            print("="* 80)
            print(f"From: {stats['time_period']['start']}")
            print(f"To:   {stats['time_period']['end']}")
        
        if 'gender_distribution' in stats:
            print("\nGender Distribution:")
            print("="* 80)
            # Format: gender(20) | count(10) | percentage(10)
            print(f"{'Gender':<20} {'Count':>10} {'Percent':>10}")
            print("-" * 80)
            total = sum(stats['gender_distribution'].values())
            for gender, count in sorted(stats['gender_distribution'].items(), key=lambda x: x[1], reverse=True):
                print(f"{gender:<20} {count:>10,d} {count/total*100:>9.1f}%")
        
        if 'ruling_party_distribution' in stats:
            print("\nRuling Party Distribution:")
            print("="* 80)
            # Format: status(20) | count(10) | percentage(10)
            print(f"{'Status':<20} {'Count':>10} {'Percent':>10}")
            print("-" * 80)
            total = sum(stats['ruling_party_distribution'].values())
            for status, count in sorted(stats['ruling_party_distribution'].items(), key=lambda x: x[1], reverse=True):
                print(f"{status:<20} {count:>10,d} {count/total*100:>9.1f}%")
        
    def prepare_embeddings_for_lda(self, 
                                min_speeches_per_party: int = 100,
                                layers: Optional[Union[int, List[int], str]] = None
                                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare embeddings for LDA analysis by filtering and encoding parties.
        """
        # Filter parties with sufficient data
        party_counts = self.df['party'].value_counts()
        valid_parties = party_counts[party_counts >= min_speeches_per_party].index
        
        # Filter dataframe and store it
        self.filtered_df = self.df[self.df['party'].isin(valid_parties)].copy()
        
        # Encode parties
        self.filtered_df['party_encoded'] = self.party_encoder.fit_transform(self.filtered_df['party'])
        
        # Process layer selection
        embeddings = np.stack(self.filtered_df['embedding'].values)
        
        if layers is None or layers == "all":
            X = embeddings
        elif isinstance(layers, int):
            X = embeddings[:, layers, :]
        elif isinstance(layers, list):
            X = embeddings[:, layers, :]
        elif layers == "last":
            X = embeddings[:, -1, :]
        elif layers == "mean":
            X = np.mean(embeddings, axis=1)
        else:
            raise ValueError(f"Invalid layers parameter: {layers}")
            
        y = self.filtered_df['party_encoded'].values
        
        return X, y
    
    def perform_lda(self, 
                    n_components: int = 3, 
                    min_speeches_per_party: int = 100,
                    layers: Optional[Union[int, List[int], str]] = None
                    ) -> Tuple[LinearDiscriminantAnalysis, np.ndarray]:
        """
        Perform LDA on the embeddings.
        
        Args:
            n_components: Number of LDA components to extract
            min_speeches_per_party: Minimum speeches required for a party
            layers: Which layers to use. Can be:
                   - int: Single layer index
                   - List[int]: List of layer indices
                   - "all": Use all layers (default)
                   - "last": Use only the last layer
                   - "mean": Take mean across all layers
            
        Returns:
            Tuple of (fitted LDA model, transformed embeddings)
        """
        X, y = self.prepare_embeddings_for_lda(min_speeches_per_party, layers)
        
        # Flatten embeddings if they're 3D (in case of multiple layers)
        if len(X.shape) > 2:
            print(f"X {X.shape}")
            X = X.reshape(X.shape[0], -1)
            print(f"X {X.shape}")
        
        # Perform LDA
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_lda = lda.fit_transform(X, y)
        
        return lda, X_lda
        
    def plot_lda_results_2d(self, X_lda: np.ndarray, layer: Optional[int] = None, save_path: str = None, n_std: float = 3.0):
        """
        Plot the LDA results in 2D scatter plot format with outlier handling.
        
        Args:
            X_lda: Transformed embeddings from LDA (n_samples, 2)
            layer: Layer number for title (optional)
            save_path: Optional path to save the plot
            n_std: Number of standard deviations for outlier removal (default: 3.0)
        """
        if X_lda.shape[1] != 2:
            raise ValueError("This plotting function requires exactly 2 LDA components")
        
        if self.filtered_df is None:
            raise ValueError("No LDA has been performed yet. Run perform_lda first.")
        
        # Create a copy of the data for outlier handling
        X_plot = X_lda.copy()
        
        # Calculate z-scores for both dimensions
        z_scores = np.abs(stats.zscore(X_plot))
        
        # Create mask for non-outlier points
        mask = (z_scores < n_std).all(axis=1)
        
        # Apply mask to both the LDA coordinates and the filtered DataFrame
        X_plot = X_plot[mask]
        plot_df = self.filtered_df[mask].copy()
        
        # Print outlier statistics
        total_points = len(mask)
        outliers = np.sum(~mask)
        print(f"Removed {outliers} outliers out of {total_points} points ({(outliers/total_points)*100:.2f}%)")
            
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Get unique parties from filtered data
        unique_parties = plot_df['party'].unique()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_parties)))
        
        # Create scatter plot for each party
        for i, party in enumerate(unique_parties):
            party_mask = plot_df['party'] == party
            plt.scatter(
                X_plot[party_mask, 0],
                X_plot[party_mask, 1],
                label=party,
                color=colors[i],
                alpha=0.6,
                s=100  # Point size
            )
        
        # Add labels and title
        plt.xlabel('First Linear Discriminant')
        plt.ylabel('Second Linear Discriminant')
        title = 'Political Party Classification (LDA)'
        if layer is not None:
            title += f' - Layer {layer}'
        plt.title(title)
        
        # Add legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout to prevent legend overlap
        plt.tight_layout()
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Add axis limits with some padding
        x_min, x_max = X_plot[:, 0].min(), X_plot[:, 0].max()
        y_min, y_max = X_plot[:, 1].min(), X_plot[:, 1].max()
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        plt.xlim(x_min - x_padding, x_max + x_padding)
        plt.ylim(y_min - y_padding, y_max + y_padding)
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        return plt.gcf(), plt.gca()


# Initialize the analyzer
analyzer = PoliticalSpeechAnalyzer(df[df['country']=='UK'])

# Get dataset statistics
analyzer.print_dataset_statistics()

#analyzer.prepare_embeddings_for_lda(layers=layer)
for layer in tqdm([4,8,12,15]):
    # Perform LDA on a single layer
    lda_model, X_lda = analyzer.perform_lda(n_components=2, layers=layer, min_speeches_per_party=20)
    # Plot the results
    fig, ax = analyzer.plot_lda_results_2d(X_lda, save_path=f'layer_{layer}_lda_2d_visualization.png')




plt.show()



# Perform LDA on multiple specific layers
lda_model, X_lda = analyzer.perform_lda(n_components=2, layers=[0, 5, 10])

# Analyze performance across different layer selections
layer_analysis = analyzer.analyze_layer_selection()
for selection, var_ratio in layer_analysis.items():
    print(f"{selection}: {var_ratio}")


# Plot the results
analyzer.plot_lda_results(X_lda, save_path='lda_visualization.png')

# Analyze layer contributions
layer_contributions = analyzer.analyze_layer_contributions(lda_model)