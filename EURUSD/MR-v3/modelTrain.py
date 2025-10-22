from model import ReversalModelMulti

DATA_PATH = "dataset.csv"
MODEL_PATH = "reversal_model_multi.pkl"

def main():
    print("[TRAIN] Data loading...")
    model = ReversalModelMulti(model_path=MODEL_PATH)

    df = model.load_data(DATA_PATH)
    print(f"[TRAIN] Dataset size: {len(df)} lines")

    print("[TRAIN] Model training...")
    model.train(df)

    print("[TRAIN] Training completed. Model saved in:", MODEL_PATH)

if __name__ == "__main__":
    main()
