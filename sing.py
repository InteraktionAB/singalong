from gradio import Interface

from singalong.inference import inference, inputs, outputs

if __name__ == "__main__":
    interface: Interface = Interface(
        fn=inference, inputs=inputs, outputs=outputs
    )  # noqa
    interface.launch(share=True)
