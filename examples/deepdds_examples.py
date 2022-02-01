"""Example with DeepDDs."""

from chemicalx import pipeline
from chemicalx.data import DrugCombDB
from chemicalx.models import DeepDDS


def main():
    """Train and evaluate the DeepDDs model."""
    dataset = DrugCombDB()

    # Run the GCN DeepDDS model
    model = DeepDDS(
        context_feature_size=dataset.context_channels,
        context_output_size=dataset.drug_channels,
        dropout=0.5,  # Rate used in source paper for DeepDDS
        drug_feature_model="gcn",
    )
    results = pipeline(
        dataset=dataset,
        model=model,
        batch_size=5120,
        epochs=1,  # FixMe Set to something more reasonable later
        context_features=True,
        drug_features=True,
        drug_molecules=True,
    )
    results.summarize()

    # Run the GAT DeepDDS model
    model = DeepDDS(
        context_feature_size=dataset.context_channels,
        context_output_size=dataset.drug_channels,
        dropout=0.5,  # Rate used in source paper for DeepDDS
        drug_feature_model="gat",
    )
    results = pipeline(
        dataset=dataset,
        model=model,
        batch_size=5120,
        epochs=1,  # FixMe Set to something more reasonable later
        context_features=True,
        drug_features=True,
        drug_molecules=True,
    )
    results.summarize()


if __name__ == "__main__":
    main()
