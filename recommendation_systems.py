"""
Recommendation Systems Cheatsheet
From collaborative filtering to deep learning approaches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

# ============================================================================
# COLLABORATIVE FILTERING - USER-BASED
# ============================================================================

def user_based_cf(user_item_matrix, user_id, k=10):
    """
    User-based collaborative filtering
    
    Expected data structure:
        user_item_matrix: numpy array of shape (n_users, n_items)
                         Each row represents a user, each column an item
                         Values are ratings (0 means not rated)
                         Example: [[5, 0, 3, 0],
                                   [4, 0, 0, 5],
                                   [0, 3, 4, 0]]
        user_id: int, index of the user to generate recommendations for
        k: int, number of similar users to consider
    
    Returns:
        List of tuples (item_id, predicted_rating) sorted by rating
    """
    # Compute user similarity
    user_similarity = cosine_similarity(user_item_matrix)
    
    # Find k most similar users
    similar_users = user_similarity[user_id].argsort()[-k-1:-1][::-1]
    
    # Get items not rated by target user
    user_ratings = user_item_matrix[user_id]
    unrated_items = np.where(user_ratings == 0)[0]
    
    # Predict ratings
    predictions = {}
    for item in unrated_items:
        # Weighted average of similar users' ratings
        similar_ratings = user_item_matrix[similar_users, item]
        similarities = user_similarity[user_id, similar_users]
        
        # Filter users who rated this item
        rated_mask = similar_ratings > 0
        if rated_mask.sum() > 0:
            weighted_sum = np.sum(similarities[rated_mask] * similar_ratings[rated_mask])
            similarity_sum = np.sum(similarities[rated_mask])
            predictions[item] = weighted_sum / similarity_sum if similarity_sum > 0 else 0
    
    # Sort by predicted rating
    recommendations = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    return recommendations

# ============================================================================
# COLLABORATIVE FILTERING - ITEM-BASED
# ============================================================================

def item_based_cf(user_item_matrix, user_id, k=10):
    """
    Item-based collaborative filtering
    
    Expected data structure:
        user_item_matrix: numpy array of shape (n_users, n_items)
                         Same format as user-based CF
                         Each row is a user, each column is an item
        user_id: int, index of the user
        k: int, number of similar items to consider
    
    Returns:
        List of tuples (item_id, predicted_rating) sorted by rating
    """
    # Compute item similarity
    item_similarity = cosine_similarity(user_item_matrix.T)
    
    # Get items rated by user
    user_ratings = user_item_matrix[user_id]
    rated_items = np.where(user_ratings > 0)[0]
    unrated_items = np.where(user_ratings == 0)[0]
    
    # Predict ratings for unrated items
    predictions = {}
    for item in unrated_items:
        # Find similar items that user has rated
        similar_scores = item_similarity[item]
        
        weighted_sum = 0
        similarity_sum = 0
        
        for rated_item in rated_items:
            sim = similar_scores[rated_item]
            if sim > 0:
                weighted_sum += sim * user_ratings[rated_item]
                similarity_sum += sim
        
        if similarity_sum > 0:
            predictions[item] = weighted_sum / similarity_sum
    
    recommendations = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    return recommendations

# ============================================================================
# MATRIX FACTORIZATION
# ============================================================================

class MatrixFactorization(nn.Module):
    """
    Basic Matrix Factorization (MF)
    
    Expected data structure:
        Training data should be tuples/tensors of (user_id, item_id, rating)
        - user_id: int tensor, shape (batch_size,)
        - item_id: int tensor, shape (batch_size,)
        - rating: float tensor, shape (batch_size,)
        
    Example:
        train_data = [(0, 5, 4.5), (1, 3, 3.0), (0, 7, 5.0)]
        # Convert to tensors and pass through DataLoader
    
    Loss: MSELoss between predicted and actual ratings
    """
    def __init__(self, num_users, num_items, embedding_dim=50):
        super(MatrixFactorization, self).__init__()
        
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
    
    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embeddings(user_ids)
        item_embeds = self.item_embeddings(item_ids)
        
        # Dot product
        predictions = (user_embeds * item_embeds).sum(dim=1)
        return predictions

class BiasedMF(nn.Module):
    """Matrix Factorization with biases"""
    def __init__(self, num_users, num_items, embedding_dim=50):
        super(BiasedMF, self).__init__()
        
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.user_biases = nn.Embedding(num_users, 1)
        self.item_biases = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.item_biases.weight)
    
    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embeddings(user_ids)
        item_embeds = self.item_embeddings(item_ids)
        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()
        
        predictions = (user_embeds * item_embeds).sum(dim=1)
        predictions = predictions + user_bias + item_bias + self.global_bias
        
        return predictions

# SVD-based approach
def svd_recommendations(user_item_matrix, n_components=50):
    """
    SVD-based collaborative filtering
    
    Expected data structure:
        user_item_matrix: numpy array or scipy sparse matrix (n_users, n_items)
                         Dense or sparse user-item interaction matrix
                         Non-zero values indicate ratings/interactions
        n_components: int, number of latent factors
    
    Returns:
        predicted_ratings: numpy array (n_users, n_items) with predicted ratings
        user_factors: numpy array (n_users, n_components)
        item_factors: numpy array (n_items, n_components)
    """
    svd = TruncatedSVD(n_components=n_components)
    user_factors = svd.fit_transform(user_item_matrix)
    item_factors = svd.components_.T
    
    # Reconstruct matrix
    predicted_ratings = np.dot(user_factors, item_factors.T)
    
    return predicted_ratings, user_factors, item_factors

# ============================================================================
# NEURAL COLLABORATIVE FILTERING (NCF)
# ============================================================================

class NeuralCF(nn.Module):
    """
    Neural Collaborative Filtering (NCF)
    
    Expected data structure:
        Input: (user_id, item_id) pairs
        - user_id: torch.LongTensor of shape (batch_size,)
        - item_id: torch.LongTensor of shape (batch_size,)
        Target: ratings or binary labels (implicit feedback)
        - ratings: torch.FloatTensor of shape (batch_size,)
        
    Example batch:
        {'user_ids': torch.tensor([0, 1, 2]),
         'item_ids': torch.tensor([5, 3, 7]),
         'ratings': torch.tensor([4.5, 3.0, 5.0])}
    
    Loss: MSELoss for explicit feedback, BCELoss for implicit feedback
    """
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_layers=[128, 64, 32]):
        super(NeuralCF, self).__init__()
        
        # Embeddings
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        layers = []
        input_dim = embedding_dim * 2
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
    
    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embeddings(user_ids)
        item_embeds = self.item_embeddings(item_ids)
        
        # Concatenate embeddings
        x = torch.cat([user_embeds, item_embeds], dim=1)
        
        # Pass through MLP
        predictions = self.mlp(x).squeeze()
        
        return predictions

class GeneralizedMF(nn.Module):
    """Generalized Matrix Factorization (GMF) component"""
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(GeneralizedMF, self).__init__()
        
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, 1)
        
    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embeddings(user_ids)
        item_embeds = self.item_embeddings(item_ids)
        
        element_product = user_embeds * item_embeds
        output = self.output_layer(element_product).squeeze()
        
        return output

class NeuMF(nn.Module):
    """Neural Matrix Factorization - combines GMF and MLP"""
    def __init__(self, num_users, num_items, mf_dim=64, mlp_dim=64, mlp_layers=[128, 64, 32]):
        super(NeuMF, self).__init__()
        
        # GMF embeddings
        self.mf_user_embeddings = nn.Embedding(num_users, mf_dim)
        self.mf_item_embeddings = nn.Embedding(num_items, mf_dim)
        
        # MLP embeddings
        self.mlp_user_embeddings = nn.Embedding(num_users, mlp_dim)
        self.mlp_item_embeddings = nn.Embedding(num_items, mlp_dim)
        
        # MLP layers
        mlp = []
        input_dim = mlp_dim * 2
        
        for hidden_dim in mlp_layers:
            mlp.append(nn.Linear(input_dim, hidden_dim))
            mlp.append(nn.ReLU())
            mlp.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        self.mlp = nn.Sequential(*mlp)
        
        # Final prediction layer
        self.predict = nn.Linear(mf_dim + mlp_layers[-1], 1)
        
    def forward(self, user_ids, item_ids):
        # GMF part
        mf_user = self.mf_user_embeddings(user_ids)
        mf_item = self.mf_item_embeddings(item_ids)
        gmf_output = mf_user * mf_item
        
        # MLP part
        mlp_user = self.mlp_user_embeddings(user_ids)
        mlp_item = self.mlp_item_embeddings(item_ids)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=1)
        mlp_output = self.mlp(mlp_input)
        
        # Concatenate and predict
        combined = torch.cat([gmf_output, mlp_output], dim=1)
        prediction = self.predict(combined).squeeze()
        
        return prediction

# ============================================================================
# CONTENT-BASED FILTERING
# ============================================================================

class ContentBasedRecommender:
    """
    Content-based recommendation using item features
    
    Expected data structure:
        item_features: pandas DataFrame or numpy array (n_items, n_features)
                      Each row represents an item
                      Columns are numerical features describing the item
                      
        Example DataFrame:
            Item_ID | Genre_Action | Genre_Comedy | Year | Rating
            0       | 1            | 0            | 2020 | 4.5
            1       | 0            | 1            | 2019 | 3.8
            2       | 1            | 1            | 2021 | 4.2
        
        Or use TF-IDF vectors for text features like descriptions
    """
    
    def __init__(self, item_features):
        """
        item_features: DataFrame with items as rows and features as columns
        """
        self.item_features = item_features
        self.item_similarity = cosine_similarity(item_features)
    
    def recommend(self, item_id, top_k=10):
        """Recommend similar items based on content"""
        similarities = self.item_similarity[item_id]
        similar_items = similarities.argsort()[-top_k-1:-1][::-1]
        
        recommendations = [(item, similarities[item]) for item in similar_items]
        return recommendations
    
    def recommend_for_user(self, user_profile, top_k=10):
        """Recommend items matching user profile"""
        # user_profile: vector of feature preferences
        similarities = cosine_similarity([user_profile], self.item_features)[0]
        top_items = similarities.argsort()[-top_k:][::-1]
        
        recommendations = [(item, similarities[item]) for item in top_items]
        return recommendations

# TF-IDF based content filtering
def tfidf_content_based(item_descriptions, item_id, top_k=10):
    """
    Content-based filtering using TF-IDF
    
    Expected data structure:
        item_descriptions: list of strings
                          Each string is a text description of an item
                          Example: ['Action movie about heroes',
                                   'Romantic comedy film',
                                   'Sci-fi thriller with robots']
        item_id: int, index of the item to find similar items for
        top_k: int, number of similar items to return
    
    Returns:
        similar_indices: numpy array of item indices
        similarity_scores: numpy array of similarity scores
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(item_descriptions)
    
    # Compute similarity
    similarity = cosine_similarity(tfidf_matrix[item_id:item_id+1], tfidf_matrix).flatten()
    similar_indices = similarity.argsort()[-top_k-1:-1][::-1]
    
    return similar_indices, similarity[similar_indices]

# ============================================================================
# HYBRID RECOMMENDER
# ============================================================================

class HybridRecommender(nn.Module):
    """
    Hybrid recommendation combining collaborative and content-based
    
    Expected data structure:
        Training batch should contain:
        - user_ids: torch.LongTensor (batch_size,)
        - item_ids: torch.LongTensor (batch_size,)
        - item_features: torch.FloatTensor (batch_size, num_content_features)
        - ratings: torch.FloatTensor (batch_size,)
        
    Example:
        {'user_ids': torch.tensor([0, 1]),
         'item_ids': torch.tensor([5, 3]),
         'item_features': torch.tensor([[1, 0, 2020, 4.5],
                                        [0, 1, 2019, 3.8]]),
         'ratings': torch.tensor([4.5, 3.0])}
    """
    def __init__(self, num_users, num_items, num_content_features, 
                 embedding_dim=64, hidden_dim=128):
        super(HybridRecommender, self).__init__()
        
        # Collaborative part
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        
        # Content-based part
        self.content_fc = nn.Linear(num_content_features, embedding_dim)
        
        # Combination layers
        self.fc1 = nn.Linear(embedding_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, user_ids, item_ids, item_features):
        # Collaborative embeddings
        user_embeds = self.user_embeddings(user_ids)
        item_embeds = self.item_embeddings(item_ids)
        
        # Content features
        content_embeds = F.relu(self.content_fc(item_features))
        
        # Combine all features
        x = torch.cat([user_embeds, item_embeds, content_embeds], dim=1)
        
        # MLP
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x).squeeze()
        
        return x

# ============================================================================
# FACTORIZATION MACHINES
# ============================================================================

class FactorizationMachine(nn.Module):
    """
    Factorization Machines for recommendation
    
    Expected data structure:
        Input: feature vector x of shape (batch_size, num_features)
               This should be a concatenation of one-hot encoded user_id,
               item_id, and any additional features
               
        Example for one sample:
            User 0, Item 5, Context features [1, 0, 3]:
            If num_users=100, num_items=1000:
            x = [1,0,0,...,0 (100 dims), 0,0,0,0,0,1,0,... (1000 dims), 1,0,3]
                 ^user one-hot      ^item one-hot              ^context features
        
        Often created using sklearn's OneHotEncoder or manual encoding
    """
    def __init__(self, num_features, embedding_dim=10):
        super(FactorizationMachine, self).__init__()
        
        # Linear terms
        self.linear = nn.Linear(num_features, 1)
        
        # Factorization terms
        self.embeddings = nn.Parameter(torch.randn(num_features, embedding_dim))
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # x: (batch, num_features)
        
        # Linear term
        linear_part = self.linear(x)
        
        # Interaction term
        # (sum of squares) - (square of sums)
        square_of_sum = torch.pow(torch.mm(x, self.embeddings), 2)
        sum_of_square = torch.mm(torch.pow(x, 2), torch.pow(self.embeddings, 2))
        interaction_part = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        
        output = linear_part + interaction_part + self.bias
        
        return output.squeeze()

class DeepFM(nn.Module):
    """Deep Factorization Machine"""
    def __init__(self, num_features, embedding_dim=10, hidden_layers=[256, 128, 64]):
        super(DeepFM, self).__init__()
        
        # FM component
        self.linear = nn.Linear(num_features, 1)
        self.embeddings = nn.Parameter(torch.randn(num_features, embedding_dim))
        
        # Deep component
        deep_layers = []
        input_dim = num_features * embedding_dim
        
        for hidden_dim in hidden_layers:
            deep_layers.append(nn.Linear(input_dim, hidden_dim))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        deep_layers.append(nn.Linear(input_dim, 1))
        self.deep = nn.Sequential(*deep_layers)
        
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # FM part
        linear_part = self.linear(x)
        
        square_of_sum = torch.pow(torch.mm(x, self.embeddings), 2)
        sum_of_square = torch.mm(torch.pow(x, 2), torch.pow(self.embeddings, 2))
        fm_part = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        
        # Deep part
        embedding_input = torch.mm(x, self.embeddings).view(x.size(0), -1)
        deep_part = self.deep(embedding_input)
        
        output = linear_part + fm_part + deep_part + self.bias
        
        return output.squeeze()

# ============================================================================
# SEQUENTIAL RECOMMENDATION
# ============================================================================

class GRU4Rec(nn.Module):
    """
    GRU for session-based recommendation
    
    Expected data structure:
        Input: item_seq of shape (batch_size, seq_len)
               Sequences of item IDs that user interacted with in order
               Use padding_idx=0 for variable-length sequences
               
        Example batch:
            [[5, 3, 7, 2, 0],   # User session: items 5→3→7→2, then padded
             [1, 4, 6, 0, 0],   # User session: items 1→4→6, then padded
             [8, 2, 3, 9, 1]]   # User session: items 8→2→3→9→1
        
        Target: next item the user will interact with (for each sequence)
        Loss: CrossEntropyLoss
    """
    def __init__(self, num_items, embedding_dim=128, hidden_dim=256, num_layers=1):
        super(GRU4Rec, self).__init__()
        
        self.embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, num_items)
        
    def forward(self, item_seq):
        # item_seq: (batch, seq_len)
        embedded = self.embedding(item_seq)
        gru_out, _ = self.gru(embedded)
        
        # Use last output
        last_output = gru_out[:, -1, :]
        logits = self.output_layer(last_output)
        
        return logits

class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation
    
    Expected data structure:
        Same as GRU4Rec:
        Input: item_seq of shape (batch_size, seq_len)
               Sequential item interactions
               
        Example:
            [[12, 45, 67, 23, 0, 0],
             [8, 91, 34, 56, 78, 12],
             [5, 23, 0, 0, 0, 0]]
        
        Target: next item ID (classification task)
        Loss: CrossEntropyLoss
        
        Better than GRU4Rec due to self-attention mechanism
    """
    def __init__(self, num_items, embedding_dim=128, num_heads=4, num_blocks=2, dropout=0.2):
        super(SASRec, self).__init__()
        
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(1000, embedding_dim)  # Max sequence length
        
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_blocks)
        ])
        
        self.output_layer = nn.Linear(embedding_dim, num_items)
        
    def forward(self, item_seq):
        # item_seq: (batch, seq_len)
        batch_size, seq_len = item_seq.size()
        
        # Embeddings
        positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand(batch_size, -1)
        item_embeds = self.item_embedding(item_seq)
        pos_embeds = self.position_embedding(positions)
        
        x = item_embeds + pos_embeds
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Output
        logits = self.output_layer(x[:, -1, :])  # Predict next item
        
        return logits

# ============================================================================
# CONTEXT-AWARE RECOMMENDATION
# ============================================================================

class ContextAwareRecommender(nn.Module):
    """
    Recommendation with contextual information (time, location, etc.)
    
    Expected data structure:
        Batch contains:
        - user_ids: torch.LongTensor (batch_size,)
        - item_ids: torch.LongTensor (batch_size,)
        - context_features: torch.FloatTensor (batch_size, num_context_features)
        - ratings: torch.FloatTensor (batch_size,) [target]
        
        Context features can include:
        - Time of day (hour normalized 0-1)
        - Day of week (one-hot encoded)
        - User location (encoded)
        - Device type (encoded)
        - Season, weather, etc.
        
        Example:
            context_features = [0.75,  # Time: 6pm (18/24)
                               0, 1, 0, 0, 0, 0, 0,  # Day: Tuesday (one-hot)
                               1, 0, 0]  # Device: mobile (one-hot)
    """
    def __init__(self, num_users, num_items, num_context_features, 
                 embedding_dim=64, hidden_dim=128):
        super(ContextAwareRecommender, self).__init__()
        
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.context_fc = nn.Linear(num_context_features, embedding_dim)
        
        self.fc1 = nn.Linear(embedding_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, user_ids, item_ids, context_features):
        user_embeds = self.user_embeddings(user_ids)
        item_embeds = self.item_embeddings(item_ids)
        context_embeds = F.relu(self.context_fc(context_features))
        
        x = torch.cat([user_embeds, item_embeds, context_embeds], dim=1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x).squeeze()
        
        return x

# ============================================================================
# GRAPH-BASED RECOMMENDATION
# ============================================================================

class LightGCN(nn.Module):
    """
    Light Graph Convolutional Network for recommendation
    
    Expected data structure:
        Input during forward:
        - user_ids: torch.LongTensor (batch_size,)
        - item_ids: torch.LongTensor (batch_size,)
        - adj_matrix: torch sparse tensor, normalized adjacency matrix
                     Shape: (num_users + num_items, num_users + num_items)
        
        Adjacency matrix construction:
            Build bipartite graph where users connect to items they interacted with
            
            Example with 3 users, 4 items:
            adj = [[0, 0, 0, 1, 0, 1, 0],  # User 0 → Items 3, 5
                   [0, 0, 0, 0, 1, 0, 1],  # User 1 → Items 4, 6
                   [0, 0, 0, 1, 1, 0, 0],  # User 2 → Items 3, 4
                   [1, 0, 1, 0, 0, 0, 0],  # Item 3 → Users 0, 2
                   [0, 1, 1, 0, 0, 0, 0],  # Item 4 → Users 1, 2
                   [1, 0, 0, 0, 0, 0, 0],  # Item 5 → User 0
                   [0, 1, 0, 0, 0, 0, 0]]  # Item 6 → User 1
            
            Then normalize using D^(-1/2) * A * D^(-1/2)
    """
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super(LightGCN, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        
        # Initialize embeddings
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        
        nn.init.normal_(self.user_embeddings.weight, std=0.1)
        nn.init.normal_(self.item_embeddings.weight, std=0.1)
        
    def forward(self, user_ids, item_ids, adj_matrix):
        # adj_matrix: normalized adjacency matrix
        
        # Initial embeddings
        all_embeddings = torch.cat([
            self.user_embeddings.weight,
            self.item_embeddings.weight
        ], dim=0)
        
        embeddings_list = [all_embeddings]
        
        # Graph convolution
        for _ in range(self.num_layers):
            all_embeddings = torch.sparse.mm(adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        
        # Average across layers
        final_embeddings = torch.stack(embeddings_list, dim=0).mean(dim=0)
        
        user_embeds = final_embeddings[user_ids]
        item_embeds = final_embeddings[self.num_users + item_ids]
        
        # Inner product
        predictions = (user_embeds * item_embeds).sum(dim=1)
        
        return predictions

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def train_recommender(model, train_loader, criterion, optimizer, 
                      num_epochs=20, device='cuda'):
    """
    Training loop for recommendation models
    
    Expected data structure:
        train_loader: PyTorch DataLoader yielding batches as dicts
                     Each batch should contain:
                     {'user_ids': tensor, 'item_ids': tensor, 'ratings': tensor}
        
        Example setup:
            from torch.utils.data import Dataset, DataLoader
            
            class RecDataset(Dataset):
                def __init__(self, users, items, ratings):
                    self.users = users
                    self.items = items
                    self.ratings = ratings
                
                def __len__(self):
                    return len(self.users)
                
                def __getitem__(self, idx):
                    return {
                        'user_ids': self.users[idx],
                        'item_ids': self.items[idx],
                        'ratings': self.ratings[idx]
                    }
            
            dataset = RecDataset(user_ids, item_ids, ratings)
            train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    criterion: torch.nn.MSELoss() for explicit feedback
              torch.nn.BCEWithLogitsLoss() for implicit feedback
    """
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            user_ids = batch['user_ids'].to(device)
            item_ids = batch['item_ids'].to(device)
            ratings = batch['ratings'].to(device)
            
            optimizer.zero_grad()
            
            predictions = model(user_ids, item_ids)
            loss = criterion(predictions, ratings)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}')
    
    return model

def train_with_bpr_loss(model, train_loader, optimizer, num_epochs=20, device='cuda'):
    """
    Training with Bayesian Personalized Ranking (BPR) loss
    Used for implicit feedback (clicks, views) where we only have positive interactions
    
    Expected data structure:
        train_loader: DataLoader yielding batches with:
                     {'user_ids': tensor,
                      'pos_item_ids': tensor (items user interacted with),
                      'neg_item_ids': tensor (items user didn't interact with)}
        
        Example batch creation:
            For each user interaction (user, positive_item):
            - Sample a negative item randomly from items user hasn't interacted with
            - Create triplet (user, positive_item, negative_item)
        
        BPR loss encourages: score(user, pos_item) > score(user, neg_item)
    """
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            user_ids = batch['user_ids'].to(device)
            pos_item_ids = batch['pos_item_ids'].to(device)
            neg_item_ids = batch['neg_item_ids'].to(device)
            
            optimizer.zero_grad()
            
            pos_scores = model(user_ids, pos_item_ids)
            neg_scores = model(user_ids, neg_item_ids)
            
            # BPR loss: -log(sigmoid(pos_score - neg_score))
            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs} - BPR Loss: {avg_loss:.4f}')
    
    return model

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_hit_ratio(predictions, ground_truth, k=10):
    """
    Calculate Hit Ratio@K
    
    Expected data structure:
        predictions: numpy array of shape (n_items,) with predicted scores
        ground_truth: list or array of item indices that user actually interacted with
        k: int, top-k items to consider
    
    Example:
        predictions = np.array([0.1, 0.8, 0.3, 0.9, 0.2])  # Scores for 5 items
        ground_truth = [1, 3]  # User actually liked items 1 and 3
        hr = calculate_hit_ratio(predictions, ground_truth, k=2)
        # Will check if any of top-2 predicted items (3, 1) are in ground_truth
    
    Returns:
        1.0 if at least one ground truth item is in top-k, else 0.0
    """
    top_k = predictions.argsort()[-k:][::-1]
    hits = len(set(top_k) & set(ground_truth))
    return 1.0 if hits > 0 else 0.0

def calculate_ndcg(predictions, ground_truth, k=10):
    """Calculate Normalized Discounted Cumulative Gain@K"""
    top_k = predictions.argsort()[-k:][::-1]
    
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)
    
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k))])
    
    return dcg / idcg if idcg > 0 else 0.0

def calculate_map(predictions, ground_truth, k=10):
    """Calculate Mean Average Precision@K"""
    top_k = predictions.argsort()[-k:][::-1]
    
    score = 0.0
    num_hits = 0.0
    
    for i, item in enumerate(top_k):
        if item in ground_truth:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    
    return score / min(len(ground_truth), k) if len(ground_truth) > 0 else 0.0

def calculate_recall(predictions, ground_truth, k=10):
    """Calculate Recall@K"""
    top_k = predictions.argsort()[-k:][::-1]
    hits = len(set(top_k) & set(ground_truth))
    return hits / len(ground_truth) if len(ground_truth) > 0 else 0.0

def calculate_precision(predictions, ground_truth, k=10):
    """Calculate Precision@K"""
    top_k = predictions.argsort()[-k:][::-1]
    hits = len(set(top_k) & set(ground_truth))
    return hits / k

def evaluate_recommender(model, test_data, k_list=[5, 10, 20], device='cuda'):
    """Comprehensive evaluation of recommender system"""
    model.eval()
    
    metrics = {f'HR@{k}': [] for k in k_list}
    metrics.update({f'NDCG@{k}': [] for k in k_list})
    metrics.update({f'Recall@{k}': [] for k in k_list})
    
    with torch.no_grad():
        for user_id, pos_items, all_items in test_data:
            # Get predictions for all items
            user_tensor = torch.tensor([user_id] * len(all_items)).to(device)
            item_tensor = torch.tensor(all_items).to(device)
            
            predictions = model(user_tensor, item_tensor).cpu().numpy()
            
            for k in k_list:
                metrics[f'HR@{k}'].append(calculate_hit_ratio(predictions, pos_items, k))
                metrics[f'NDCG@{k}'].append(calculate_ndcg(predictions, pos_items, k))
                metrics[f'Recall@{k}'].append(calculate_recall(predictions, pos_items, k))
    
    # Average metrics
    for key in metrics:
        metrics[key] = np.mean(metrics[key])
    
    return metrics

# ============================================================================
# COLD START PROBLEM
# ============================================================================

def handle_cold_start_user(user_features, item_embeddings, k=10):
    """Handle cold start for new users using content features"""
    # Use user features to find similar users or recommend popular items
    # This is a simple approach - recommend most popular items
    
    item_popularity = item_embeddings.sum(dim=0)  # Simple popularity score
    top_items = item_popularity.argsort(descending=True)[:k]
    
    return top_items

def handle_cold_start_item(item_features, user_item_matrix, k=10):
    """Handle cold start for new items using content similarity"""
    # Find similar items based on content features
    similarity_scores = cosine_similarity([item_features], user_item_matrix.T)[0]
    similar_items = similarity_scores.argsort()[-k:][::-1]
    
    return similar_items
