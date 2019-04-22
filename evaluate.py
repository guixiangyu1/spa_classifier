from model.data_utils import CoNLLDataset, get_processing_word, CoNLLdata4classifier
from model.ner_model import NERModel
from model.config import Config



def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build("train")
    model.restore_session(config.dir_model)

    # create dataset
    processing_word = get_processing_word(lowercase=True)

    test = CoNLLDataset(config.filename_test, processing_word)


    test4cl = CoNLLdata4classifier(test, processing_word=config.processing_word,
                                   processing_tag=config.processing_tag)

    # evaluate and interact
    model.evaluate(test4cl)
    # interactive_shell(model)


if __name__ == "__main__":
    main()
