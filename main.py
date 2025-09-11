#!/usr/bin/env python3
"""
AI-Driven Deep-Sea Biodiversity Assessment from eDNA
====================================================

This script implements a complete AI pipeline for analyzing deep-sea environmental DNA (eDNA)
to assess biodiversity without relying on incomplete external databases. The approach uses
Variational Autoencoders (VAE) and unsupervised clustering to identify both known and novel taxa.

Key Components:
1. Mock eDNA data generation (simulates bioinformatics preprocessing output)
2. k-mer vectorization for numerical representation of sequences
3. Variational Autoencoder for learning genetic structure embeddings
4. DBSCAN clustering for taxonomic classification
5. Biodiversity metrics calculation
6. Structured JSON output

Author: AI-Driven Biodiversity Assessment Team
Date: 2025
"""

import numpy as np
import pandas as pd
import json
import random
import string
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
import os
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend suitable for servers
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class VAE(keras.Model):
    """
    A corrected Variational Autoencoder model for unsupervised clustering.
    The VAE learns a compressed, continuous representation of the genetic sequences
    that captures their underlying structure, crucial for identifying patterns.
    """
    def __init__(self, original_dim, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = keras.Sequential(
            [
                keras.Input(shape=(original_dim,)),
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(64, activation="relu"),
                layers.Dense(latent_dim + latent_dim),  # Mean and log_var
            ]
        )
        self.decoder = keras.Sequential(
            [
                keras.Input(shape=(latent_dim,)),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(256, activation="relu"),
                layers.Dense(original_dim, activation="sigmoid"), # Reconstruction
            ]
        )
    
    def call(self, inputs):
        """Forward pass through the VAE."""
        z_mean, z_log_var = tf.split(self.encoder(inputs), num_or_size_splits=2, axis=1)
        z = self.reparameterize(z_mean, z_log_var)
        reconstruction = self.decoder(z)
        
        # Calculate and add KL divergence loss directly in the call method
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        self.add_loss(kl_loss)
        return reconstruction
    
    def reparameterize(self, z_mean, z_log_var):
        """The reparameterization trick for VAEs."""
        batch = tf.shape(z_mean)[0]
        dim = self.latent_dim
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def get_embeddings(self, inputs):
        """
        Extract the latent space embeddings (z_mean) from the encoder.
        This is what is used for clustering.
        """
        z_mean, _ = tf.split(self.encoder(inputs), num_or_size_splits=2, axis=1)
        return z_mean

class eDNABiodiversityPipeline:
    """
    Complete pipeline for AI-driven biodiversity assessment from eDNA sequences.
    
    This class encapsulates the entire workflow from raw sequence generation through
    AI-based analysis to biodiversity metrics calculation.
    """
    
    def __init__(self, n_sequences=20000, sequence_length=200, k=4):
        """
        Initialize the pipeline with key parameters.
        
        Args:
            n_sequences (int): Number of mock eDNA sequences to generate
            sequence_length (int): Length of each DNA sequence
            k (int): k-mer length for sequence vectorization
        """
        self.n_sequences = n_sequences
        self.sequence_length = sequence_length
        self.k = k
        self.sequences = None
        self.kmer_vectors = None
        self.vae_model = None
        self.embeddings = None
        self.clusters = None
        self.results = {}
        # Ensure plots directory exists (aligned with Flask app's /plots route)
        self.plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
    def generate_mock_edna_sequences(self):
        """
        Generate mock eDNA sequences to simulate the output of bioinformatics preprocessing.
        
        In a real-world scenario, this would be replaced by actual ASV (Amplicon Sequence Variants)
        data from quality-filtered, denoised sequencing reads. The mock data includes:
        - Realistic base composition (A, T, G, C)
        - Variable sequence patterns to simulate different taxa
        - Some sequences with higher similarity (same taxa) and others more divergent (different taxa)
        
        Returns:
            list: Mock DNA sequences as strings
        """
        print("Generating mock eDNA sequences...")
        
        sequences = []
        bases = ['A', 'T', 'G', 'C']
        
        # Create some "template" sequences to simulate related taxa
        # In real data, related species would share sequence similarity
        templates = []
        for i in range(50):  # Create 50 base templates
            template = ''.join(np.random.choice(bases, size=self.sequence_length))
            templates.append(template)
        
        # Generate sequences with controlled similarity patterns
        for i in range(self.n_sequences):
            if i < self.n_sequences * 0.7:  # 70% of sequences are variations of templates
                # Select a random template and introduce mutations
                base_template = random.choice(templates)
                sequence = list(base_template)
                
                # Introduce random mutations (5-15% of positions)
                n_mutations = np.random.randint(int(0.05 * self.sequence_length), 
                                                int(0.15 * self.sequence_length))
                mutation_positions = np.random.choice(self.sequence_length, 
                                                      size=n_mutations, replace=False)
                
                for pos in mutation_positions:
                    sequence[pos] = np.random.choice(bases)
                
                sequences.append(''.join(sequence))
            else:  # 30% completely random (novel/distant taxa)
                sequence = ''.join(np.random.choice(bases, size=self.sequence_length))
                sequences.append(sequence)
        
        self.sequences = sequences
        print(f"Generated {len(sequences)} mock eDNA sequences of length {self.sequence_length}")
        return sequences
    
    def sequences_to_kmers(self, sequences):
        """
        Convert DNA sequences to k-mer frequency vectors for numerical analysis.
        
        k-mers are substrings of length k that provide a way to numerically represent
        genetic sequences while preserving important sequence information. This is
        crucial for machine learning approaches.
        
        Args:
            sequences (list): DNA sequences as strings
            
        Returns:
            numpy.ndarray: Matrix of k-mer frequency vectors
        """
        print(f"Converting sequences to {self.k}-mer vectors...")
        
        # Generate all possible k-mers
        bases = ['A', 'T', 'G', 'C']
        from itertools import product
        all_kmers = [''.join(p) for p in product(bases, repeat=self.k)]
        kmer_to_index = {kmer: i for i, kmer in enumerate(all_kmers)}
        
        # Initialize k-mer count matrix
        kmer_matrix = np.zeros((len(sequences), len(all_kmers)))
        
        for seq_idx, sequence in enumerate(sequences):
            # Count k-mers in this sequence
            kmer_counts = Counter()
            for i in range(len(sequence) - self.k + 1):
                kmer = sequence[i:i + self.k]
                if all(base in bases for base in kmer):  # Valid k-mer
                    kmer_counts[kmer] += 1
            
            # Convert counts to frequencies and populate matrix
            total_kmers = sum(kmer_counts.values())
            if total_kmers > 0:
                for kmer, count in kmer_counts.items():
                    if kmer in kmer_to_index:
                        kmer_matrix[seq_idx, kmer_to_index[kmer]] = count / total_kmers
        
        print(f"Created k-mer matrix of shape {kmer_matrix.shape}")
        return kmer_matrix
    
    def train_vae(self, kmer_vectors, epochs=30):
        """
        Train the VAE on k-mer vectors to learn sequence embeddings.
        
        In a real-world scenario, this would require much more computational power
        and potentially thousands of epochs. The training learns to compress the
        high-dimensional k-mer space into a meaningful lower-dimensional representation.
        
        Args:
            kmer_vectors (numpy.ndarray): k-mer frequency vectors
            epochs (int): Number of training epochs
        
        Returns:
            numpy.ndarray: Learned embeddings from the VAE encoder
        """
        print("Training VAE model...")
        
        # Standardize the input data
        scaler = StandardScaler()
        kmer_vectors_scaled = scaler.fit_transform(kmer_vectors)
        
        # Build the VAE
        input_dim = kmer_vectors_scaled.shape[1]
        self.vae_model = VAE(original_dim=input_dim)
        
        # Train the VAE
        print(f"VAE model built with input dim: {input_dim}, latent dim: {self.vae_model.latent_dim}")
        
        self.vae_model.compile(optimizer=keras.optimizers.Adam())
        self.vae_model.fit(
            kmer_vectors_scaled,
            kmer_vectors_scaled,
            epochs=epochs,
            batch_size=128,
            validation_split=0.2,
            verbose=1
        )
        
        # Extract embeddings from trained encoder
        embeddings = self.vae_model.get_embeddings(kmer_vectors_scaled)
        self.embeddings = embeddings.numpy()
        self.scaler = scaler  # Save scaler for potential future use
        
        print(f"VAE training completed. Embeddings shape: {self.embeddings.shape}")
        return self.embeddings
    
    def cluster_embeddings(self, embeddings, eps=0.5, min_samples=5):
        """
        Perform DBSCAN clustering on VAE embeddings to identify taxonomic groups.
        
        DBSCAN is particularly well-suited for this application because:
        1. It can identify outliers (noise points) as potential novel taxa
        2. It doesn't require pre-specifying the number of clusters
        3. It can find clusters of arbitrary shape
        4. Noise points represent sequences that don't fit well into any cluster,
           which may indicate novel or highly divergent taxa
        
        Args:
            embeddings (numpy.ndarray): VAE-learned sequence embeddings
            eps (float): Maximum distance between points in the same cluster
            min_samples (int): Minimum number of points to form a cluster
            
        Returns:
            numpy.ndarray: Cluster labels (-1 indicates noise/outliers)
        """
        print("Performing DBSCAN clustering on embeddings...")
        
        # Standardize embeddings for clustering
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        cluster_labels = dbscan.fit_predict(embeddings_scaled)
        
        # Analyze clustering results
        unique_clusters = set(cluster_labels)
        n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"DBSCAN clustering completed:")
        print(f"  - Number of clusters: {n_clusters}")
        print(f"  - Number of noise points (potential novel taxa): {n_noise}")
        
        if n_clusters > 1:
            silhouette = silhouette_score(embeddings_scaled, cluster_labels)
            print(f"  - Silhouette score: {silhouette:.4f}")
        else:
            print("  - Silhouette score: N/A (requires more than one cluster)")
        
        self.clusters = cluster_labels
        return cluster_labels
    
    def analyze_biodiversity(self, cluster_labels):
        """
        Calculate biodiversity metrics from clustering results.
        
        This function simulates the biological interpretation of clustering results:
        - Each cluster represents a potential taxon (species/genus/family)
        - Noise points represent potentially novel taxa not in existing databases
        - Standard ecological metrics are calculated
        
        Args:
            cluster_labels (numpy.ndarray): Cluster assignments from DBSCAN
            
        Returns:
            dict: Comprehensive biodiversity analysis results
        """
        print("Analyzing biodiversity metrics...")
        
        # Basic cluster statistics
        unique_clusters = set(cluster_labels)
        
        # Create taxa classification
        taxa_list = []
        cluster_counts = Counter(cluster_labels)
        
        # Create mock classifications for identified taxa
        mock_classifications = [
            "Eukaryota; Protista; Ciliophora", "Eukaryota; Metazoa; Chordata; Fish",
            "Eukaryota; Metazoa; Echinodermata", "Eukaryota; Metazoa; Mollusca",
            "Eukaryota; Cnidaria; Hydrozoa", "Eukaryota; Metazoa; Annelida",
            "Eukaryota; Porifera; Demospongiae", "Eukaryota; Arthropoda; Crustacea",
            "Eukaryota; Nematoda; Secernentea", "Eukaryota; Fungi; Ascomycota"
        ]
        
        # Process regular clusters (identified taxa)
        for cluster_id in sorted(unique_clusters):
            count = cluster_counts[cluster_id]
            
            if cluster_id == -1:
                # Noise points are handled as novel taxa
                continue
            
            # Simulate taxonomic assignment with mock confidence
            classification = random.choice(mock_classifications)
            
            taxon = {
                'cluster_id': int(cluster_id),
                'abundance': int(count),
                'status': 'Identified',
                'confidence': round(random.uniform(0.6, 0.95), 3),
                'taxon_name': f'{classification.split(";")[-1]}_cluster_{cluster_id}',
                'taxonomic_level': 'Species' if random.random() > 0.3 else 'Genus'
            }
            taxa_list.append(taxon)
        
        # Process noise points (novel taxa)
        n_noise = cluster_counts.get(-1, 0)
        if n_noise > 0:
            for i in range(max(1, n_noise // 50)): # Group noise points into several novel taxa
                sequences_in_taxon = max(1, n_noise // max(1, n_noise // 50))
                
                taxon = {
                    'cluster_id': f'Novel_{i+1}',
                    'abundance': int(sequences_in_taxon),
                    'status': 'Novel',
                    'confidence': 0.0,
                    'taxon_name': f'Novel_taxon_{i+1}',
                    'taxonomic_level': 'Unknown'
                }
                taxa_list.append(taxon)

        # Calculate biodiversity metrics
        abundances = [taxon['abundance'] for taxon in taxa_list]
        total_sequences = sum(abundances)
        
        # Species richness (number of taxa)
        species_richness = len(taxa_list)
        
        # Shannon Diversity Index
        shannon_index = 0
        if total_sequences > 0 and species_richness > 1:
            for abundance in abundances:
                if abundance > 0:
                    p_i = abundance / total_sequences
                    shannon_index -= p_i * np.log(p_i)
        
        # Simpson's Diversity Index
        simpson_index = 0
        if total_sequences > 0:
            for abundance in abundances:
                if abundance > 0:
                    p_i = abundance / total_sequences
                    simpson_index += p_i ** 2
        simpson_index = 1 - simpson_index
        
        # Pielou's Evenness Index
        pielou_evenness = shannon_index / np.log(species_richness) if species_richness > 1 else 0
        
        # Compile results
        results = {
            'pipeline_info': {
                'total_sequences_analyzed': int(total_sequences),
                'clustering_algorithm': 'DBSCAN',
                'embedding_method': 'Variational Autoencoder',
                'k_mer_size': self.k
            },
            'taxonomic_summary': {
                'total_taxa_identified': int(species_richness),
                'identified_taxa': int(len([t for t in taxa_list if t['status'] == 'Identified'])),
                'novel_taxa': int(len([t for t in taxa_list if t['status'] == 'Novel'])),
                'total_clusters': len(set(c for c in cluster_labels if c != -1)),
                'noise_points': int(n_noise)
            },
            'biodiversity_metrics': {
                'species_richness': int(species_richness),
                'shannon_diversity_index': round(shannon_index, 4),
                'simpson_diversity_index': round(simpson_index, 4),
                'pielou_evenness_index': round(pielou_evenness, 4)
            },
            'taxa_details': sorted(taxa_list, key=lambda x: x['abundance'], reverse=True)
        }
        
        print(f"Biodiversity analysis completed:")
        print(f"  - Total taxa: {species_richness}")
        print(f"  - Shannon diversity: {shannon_index:.4f}")
        print(f"  - Novel taxa discovered: {results['taxonomic_summary']['novel_taxa']}")
        
        return results
    
    def run_full_pipeline(self):
        """
        Execute the complete AI-driven biodiversity assessment pipeline.
        
        This orchestrates all steps from data generation through final analysis,
        simulating a real-world application for deep-sea eDNA biodiversity assessment.
        
        Returns:
            dict: Complete analysis results in JSON format
        """
        print("="*60)
        print("AI-DRIVEN DEEP-SEA BIODIVERSITY ASSESSMENT PIPELINE")
        print("="*60)
        print()
        
        # Step 1: Generate mock eDNA data
        print("STEP 1: Data Generation and Preprocessing")
        print("-" * 40)
        sequences = self.generate_mock_edna_sequences()
        
        # Step 2: Convert to numerical representation
        print("\nSTEP 2: Sequence Vectorization")
        print("-" * 40)
        self.kmer_vectors = self.sequences_to_kmers(sequences)
        
        # Step 3: Train VAE for embedding learning
        print("\nSTEP 3: Variational Autoencoder Training")
        print("-" * 40)
        embeddings = self.train_vae(self.kmer_vectors, epochs=30)
        
        # Step 4: Cluster embeddings to identify taxa
        print("\nSTEP 4: Unsupervised Taxonomic Clustering")
        print("-" * 40)
        cluster_labels = self.cluster_embeddings(embeddings)
        
        # Step 5: Analyze biodiversity
        print("\nSTEP 5: Biodiversity Assessment")
        print("-" * 40)
        results = self.analyze_biodiversity(cluster_labels)
        
        self.results = results
        # Generate and save visualization images expected by the frontend
        try:
            self._save_visualizations()
        except Exception as plot_err:
            # Do not fail the whole pipeline if plotting fails; just log
            print(f"Visualization generation failed: {plot_err}")
        
        print("\n" + "="*60)
        print("PIPELINE EXECUTION COMPLETED")
        print("="*60)
        
        return results

    def _save_visualizations(self):
        """
        Create and save visualization images used by the frontend dashboard.
        - latent_space_clusters.png: 2D projection of embeddings colored by cluster
        - kmer_heatmap.png: Heatmap of mean k-mer frequencies per cluster (top features)
        """
        if self.embeddings is None or self.clusters is None or self.kmer_vectors is None:
            raise ValueError("Required data not available to generate visualizations.")

        # 1) Latent space clusters: reduce to 2D if needed (PCA) and color by cluster
        from sklearn.decomposition import PCA

        embeddings_array = np.asarray(self.embeddings)
        if embeddings_array.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            emb_2d = pca.fit_transform(embeddings_array)
        else:
            emb_2d = embeddings_array[:, :2]

        plt.figure(figsize=(8, 6))
        unique_labels = sorted(set(self.clusters))
        palette = sns.color_palette('tab10', n_colors=max(10, len(unique_labels)))
        label_to_color = {lab: palette[idx % len(palette)] for idx, lab in enumerate(unique_labels)}
        for lab in unique_labels:
            mask = (self.clusters == lab)
            label_name = 'Noise' if lab == -1 else f'Cluster {lab}'
            plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=6, alpha=0.7, label=label_name, c=[label_to_color[lab]])
        plt.title('VAE Latent Space Clusters (PCA 2D)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        # Limit legend size for many clusters
        if len(unique_labels) <= 10:
            plt.legend(markerscale=3, fontsize=8, frameon=False)
        plt.tight_layout()
        latent_path = os.path.join(self.plots_dir, 'latent_space_clusters.png')
        plt.savefig(latent_path, dpi=150)
        plt.close()

        # 2) K-mer heatmap: compute mean k-mer frequency per cluster for top features
        kmer_matrix = np.asarray(self.kmer_vectors)
        cluster_labels = np.asarray(self.clusters)
        # Exclude noise for mean calculation but keep if all are noise
        valid_mask = cluster_labels != -1
        if valid_mask.any():
            clusters_for_stats = np.unique(cluster_labels[valid_mask])
        else:
            clusters_for_stats = np.unique(cluster_labels)

        means_by_cluster = []
        row_labels = []
        for lab in clusters_for_stats:
            mask = (cluster_labels == lab)
            if not mask.any():
                continue
            means_by_cluster.append(kmer_matrix[mask].mean(axis=0))
            row_labels.append(f'C{lab}')
        if not means_by_cluster:
            # Fallback: overall mean
            means_by_cluster.append(kmer_matrix.mean(axis=0))
            row_labels.append('All')

        means = np.vstack(means_by_cluster)
        # Pick top features by variance across clusters to make heatmap informative
        variances = means.var(axis=0)
        top_k = min(50, means.shape[1])
        top_idx = np.argsort(variances)[-top_k:]
        means_top = means[:, top_idx]

        plt.figure(figsize=(max(8, top_k * 0.25), 6))
        sns.heatmap(means_top, cmap='viridis', cbar_kws={'label': 'Mean k-mer freq'})
        plt.yticks(np.arange(len(row_labels)) + 0.5, row_labels, rotation=0)
        plt.xticks([])  # Too many features to label cleanly
        plt.title('K-mer Frequency Heatmap (cluster means, top-variable features)')
        plt.tight_layout()
        heatmap_path = os.path.join(self.plots_dir, 'kmer_heatmap.png')
        plt.savefig(heatmap_path, dpi=150)
        plt.close()

def main():
    """
    Main execution function for the AI-driven biodiversity assessment pipeline.
    
    This demonstrates the complete workflow that would be used in a real-world
    deep-sea biodiversity study, albeit with mock data and simplified parameters.
    """
    
    # Initialize the pipeline
    # In production, these parameters would be optimized based on:
    # - Available computational resources
    # - Sequencing depth and quality
    # - Target taxonomic resolution
    pipeline = eDNABiodiversityPipeline(
        n_sequences=20000,  # Number of eDNA sequences
        sequence_length=200,  # Base pairs per sequence
        k=4  # k-mer size for vectorization
    )
    
    # Execute the complete pipeline
    results = pipeline.run_full_pipeline()
    
    # Output results as formatted JSON
    print("\n" + "="*60)
    print("FINAL RESULTS (JSON FORMAT)")
    print("="*60)
    print(json.dumps(results, indent=2))
    
    # Additional insights for researchers
    print("\n" + "="*60)
    print("RESEARCH INSIGHTS")
    print("="*60)
    print(f"• Database-independent approach identified {results['taxonomic_summary']['novel_taxa']} potentially novel taxa")
    print(f"• Shannon diversity index of {results['biodiversity_metrics']['shannon_diversity_index']:.4f} indicates {'high' if results['biodiversity_metrics']['shannon_diversity_index'] > 2.0 else 'moderate'} biodiversity")
    print(f"• {results['taxonomic_summary']['noise_points']} sequences didn't cluster, suggesting unique/rare taxa")
    print(f"• Pipeline processed {results['pipeline_info']['total_sequences_analyzed']:,} sequences successfully")
    
    print("\nNOTE: This demonstration uses mock data. In production:")
    print("- Use actual eDNA sequencing data (FASTQ/FASTA files)")
    print("- Implement quality control and denoising steps")
    print("- Scale VAE architecture for larger datasets")
    print("- Optimize hyperparameters through cross-validation")
    print("- Validate results through phylogenetic analysis")

if __name__ == "__main__":
    main()
