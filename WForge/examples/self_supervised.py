"""Run from the repository root after the documented isolated backend setup."""

from wvision import Vision, compare_pretraining, pretrain


def main() -> None:
    data = ".wvision/ssl-budget/dataset.yaml"
    representation = pretrain(data, model="models/representation/simsiam-r50/smoke.yaml")
    comparison = compare_pretraining(
        data,
        pretraining_run=representation.artifact_dir,
        model="models/detection/faster-rcnn-r50/transfer-smoke.yaml",
    )
    model = Vision.from_run(comparison.pretrained_run)
    print(comparison.report)
    print(model.predict(".wvision/ssl-budget/test/images").predictions)


if __name__ == "__main__":
    main()
