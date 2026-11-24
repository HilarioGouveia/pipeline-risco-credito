from src.fase1_eda import run_eda
from src.fase2_preprocessing import run_structural_cleaning
from src.fase3_feature_engineering import run_feature_engineering
from src.fase4_data_split import prepare_modeling_datasets
from src.fase5_modelagem import train_and_benchmark_models
from src.fase6_avaliacao import generate_executive_evaluation


def execute_full_pipeline() -> dict:
    eda = run_eda()
    silver = run_structural_cleaning(eda["df"], persist=True)
    gold = run_feature_engineering(silver["df_silver"], persist=True)
    split_data = prepare_modeling_datasets(gold["df_gold"])
    modeling = train_and_benchmark_models(split_data)
    evaluation = generate_executive_evaluation(
        modeling, split_data=split_data, persist_best=True
    )

    return {
        "eda": eda,
        "silver": silver,
        "gold": gold,
        "split_data": split_data,
        "modeling": modeling,
        "evaluation": evaluation,
    }


if __name__ == "__main__":
    results = execute_full_pipeline()
    print("=== PIPELINE CONCLUÍDO COM SUCESSO ===")
    print("\nResultados Fase 5 (Treino vs Teste):")
    print(results["modeling"]["results_df"].to_string(index=False))
    print("\nComparativo Executivo Fase 6:")
    print(results["evaluation"]["comparison_df"].to_string(index=False))
    print("\nVeredito:", results["evaluation"]["verdict"]["recommended_model"])
    print("Bundle persistido em:", results["evaluation"]["bundle_path"])
