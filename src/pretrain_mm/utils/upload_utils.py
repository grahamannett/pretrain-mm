from huggingface_hub import ModelCard, ModelCardData

from pretrain_mm import logger


def make_model_card(
    base_model: str,
    model_id: str = None,
    datasets: list[str] = None,
    text: str = "",
    **kwargs,
) -> ModelCard:
    """
    Creates a model card for a given base model.

    Args:
        base_model (str): The name of the base model.
        model_id (str, optional): The ID of the model. If not provided, it will be generated from the base model name.
        datasets (list[str], optional): A list of dataset names associated with the model.
        text (str, optional): Additional text to be included in the model card.
        **kwargs: Additional keyword arguments to be passed to the ModelCard.from_template method.

    Returns:
        ModelCard: The generated model card.
    """

    if not model_id:
        model_id = f"clippy-{base_model}"
        logger.warning("model_id is not provided, made model_id from base_model")

    # card data will hold all the dataset/eval/etc stuff
    card_data = ModelCardData(
        language="en",
        library="transformers",
        datasets=datasets,
        base_model=base_model,
    )

    card = ModelCard.from_template(
        card_data,
        model_id=model_id,
        **kwargs,
    )
    if text == "":
        # card has way too much boilerplate text atm
        card.text = text
    return card
