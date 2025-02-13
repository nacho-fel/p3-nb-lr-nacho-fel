import torch
from collections import Counter
from typing import Dict

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class NaiveBayes:
    def __init__(self):
        """
        Initializes the Naive Bayes classifier
        """
        self.class_priors: Dict[int, torch.Tensor] = None
        self.conditional_probabilities: Dict[int, torch.Tensor] = None
        self.vocab_size: int = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, delta: float = 1.0):
        """
        Trains the Naive Bayes classifier by initializing class priors and estimating conditional probabilities.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.
        """
        # Estimate class priors
        self.class_priors = self.estimate_class_priors(labels)

        # Set vocabulary size
        self.vocab_size = features.shape[1]  # Number of words in the vocabulary

        # Estimate conditional probabilities
        self.conditional_probabilities = self.estimate_conditional_probabilities(features, labels, delta)


    def estimate_class_priors(self, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        Args:
            labels (torch.Tensor): Labels corresponding to each training example.

        Returns:
            Dict[int, torch.Tensor]: A dictionary mapping class labels to their estimated prior probabilities.
        """
        class_priors: Dict[int, torch.Tensor] = {}

        class_counts = torch.bincount(labels.to(torch.int64))  # Ensure labels are integers
        total_samples = labels.shape[0]

        for i in range(len(class_counts)):
            class_priors[i] = class_counts[i].item() / total_samples  # Use 'i' as key and convert tensor to Python float

        return class_priors



    def estimate_conditional_probabilities(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float
    ) -> Dict[int, torch.Tensor]:
        """
        Estimates conditional probabilities of words given a class using Laplace smoothing.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.

        Returns:
            Dict[int, torch.Tensor]: Conditional probabilities of each word for each class.
        """

        num_classes = torch.max(labels) + 1 # Number of classes
        num_words = features.shape[1]  # The number of words (vocabulary size)

        # Initialize word count matrix for each class
        class_word_counts = [torch.zeros(num_words) for _ in range(num_classes)] 

        class_counts = torch.zeros(num_classes)  # Count of total words per class

        # Count words per class
        for i in range(features.shape[0]):  # For each training example
            class_idx = labels[i].item()
            class_word_counts[class_idx] += features[i]  # stores the word counts for each class (accumulating the word counts across all examples in the same class)
            class_counts[class_idx] += torch.sum(features[i])  # Sum up words in the class

        # Apply Laplace smoothing and compute conditional probabilities
        conditional_probs = {}
        for class_idx in range(num_classes):
            total_words_in_class = class_counts[class_idx]  # Total words in class
            word_probs = (class_word_counts[class_idx] + delta) / (total_words_in_class + delta * num_words)  # Laplace smoothing (PARA TODO EL TENSOR)
            conditional_probs[class_idx] = word_probs

        return conditional_probs

    def estimate_class_posteriors(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the class posteriors for a given feature using the Naive Bayes logic.

        Args:
            feature (torch.Tensor): The bag of words vector for a single example.

        Returns:
            torch.Tensor: Log posterior probabilities for each class.
        """
        if self.conditional_probabilities is None or self.class_priors is None:
            raise ValueError(
                "Model must be trained before estimating class posteriors."
            )
        # TODO: Calculate posterior based on priors and conditional probabilities of the words
        log_posteriors: torch.Tensor = torch.zeros(len(self.class_priors))

        for i in range(len(self.class_priors)):
            log_posterior = torch.log(torch.tensor(self.class_priors[i]))

            for indx, count in enumerate(feature):  # Returns the indx, with its value 
                if count > 0:
                    log_posterior += count*torch.log(self.conditional_probabilities[i][indx])

            log_posteriors[i] = log_posterior

        return log_posteriors

    def predict(self, feature: torch.Tensor) -> int:
        """
        Classifies a new feature using the trained Naive Bayes classifier.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example to classify.

        Returns:
            int: The predicted class label (0 or 1 in binary classification).

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")
        
        # TODO: Calculate log posteriors and obtain the class of maximum likelihood 
        pred: int = None

        los_posteriors = self.estimate_class_posteriors(feature)

        pred = pred = torch.argmax(los_posteriors).item()

        return pred

    def predict_proba(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution over classes for a given feature vector.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example.

        Returns:
            torch.Tensor: A tensor representing the probability distribution over all classes.

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")

        # TODO: Calculate log posteriors and transform them to probabilities (softmax)
        probs: torch.Tensor = torch.zeros(len(self.class_priors))

        los_posteriors = self.estimate_class_posteriors(feature)

        probs = torch.softmax(los_posteriors, dim=0)

        return probs
