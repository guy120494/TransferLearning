from transfer.models.base_model_builder import build_base_model


def main():
    base_model = build_base_model()

    base_model.save('my-model.h5')


if __name__ == '__main__':
    main()
