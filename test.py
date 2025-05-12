import traceback

from inference.model_inferencer import ModelInferencer
from models import create_model
from options.test_options import TestOptions
from utils.utils import setup_logging, cleanup


def main():
    logger = setup_logging()
    logger.info("Starting test script...")

    try:
        opt = TestOptions().parse()

        if not hasattr(opt, "patch_size") or any(dim < 64 for dim in opt.patch_size):
            logger.info(
                "Setting patch size to match training configuration (200x200x20)"
            )
            opt.patch_size = [200, 200, 20]
            opt.stride_inplane = 100
            opt.stride_layer = 10

        model = create_model(opt)
        model.setup(opt)

        engine = ModelInferencer(model)

        logger.info(
            f"Starting inference with patch size {opt.patch_size}, stride: {opt.stride_inplane}, {opt.stride_layer}"
        )
        engine.run_inference(
            image_path=opt.image,
            result_path=opt.result,
            patch_size=opt.patch_size,
            stride_inplane=opt.stride_inplane,
            stride_layer=opt.stride_layer,
            batch_size=opt.batch_size,
        )
        logger.info("Inference completed successfully")

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        raise

    finally:
        cleanup()
        logger.info("Test script finished")


if __name__ == "__main__":
    main()
