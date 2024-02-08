<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill BioCLIP Module

This repository contains the code supporting the BioCLIP base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[BioCLIP](https://github.com/Imageomics/BioCLIP) is a CLIP model trained on the [TreeOfLife-10M](https://huggingface.co/datasets/imageomics/TreeOfLife-10M) dataset, created by the researchers who made BioCLIP. The dataset on which BioCLIP was trained included more than 450,000 classes.

You can use BioCLIP to auto-label natural organisms (i.e. animals, plants) in images for use in training a classification model. You can combine this model with a grounded detection model to identify the exact region in which a given class is present in an image. [Learn more about combining models with Autodistill](https://docs.autodistill.com/utilities/combine-models/).

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [BioCLIP Autodistill documentation](https://autodistill.github.io/autodistill/base_models/bioclip/).

## Installation

To use BioCLIP with autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-bioclip
```

## Quickstart

```python
from autodistill_bioclip import BioCLIP

# define an ontology to map class names to our BioCLIP prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
classes = ["arabica", "robusta"]

base_model = BioCLIP(
    ontology=CaptionOntology(
        {
            item: item for item in classes
        }
    )
)

results = base_model.predict("../arabica.jpeg")

top = results.get_top_k(1)
top_class = classes[top[0][0]]

print(f"Predicted class: {top_class}")
```


## License

This project is licensed under an [MIT license](LICENSE).

The underlying [BioCLIP model](https://huggingface.co/imageomics/bioclip) is also licensed under an MIT license.

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!