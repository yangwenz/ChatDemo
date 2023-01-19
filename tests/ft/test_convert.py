from utils.gptj_converter import convert


def test():
    saved_dir = "/home/ywz/data/models/gpt-j-6B-ft"
    in_file = "/home/ywz/data/models/gpt-j-6B"
    weight_data_type = "fp16"

    convert(
        saved_dir=saved_dir,
        in_file=in_file,
        weight_data_type=weight_data_type
    )


if __name__ == "__main__":
    test()
