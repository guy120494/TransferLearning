from transfer.models.base_model_builder import build_base_model


def main():
    models = build_base_model()

    models.save('my-model.h5')


if __name__ == '__main__':
    main()
