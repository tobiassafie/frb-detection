# Run on EC2 (needs a C/C++ compiler next to Vitis -- not available on the
# machine this project was generated on). Bit-accurate NumPy emulation of the
# quantized model, no Vitis invoked. Rebuilds the hls4ml project in-place from
# keras_model.keras (rather than trusting paths recorded in hls4ml_config.yml,
# which are relative to the original repo layout and not portable) so this also
# self-heals against any hls4ml version difference between here and EC2.
import numpy as np
import tensorflow as tf
import hls4ml

PROJECT_NAME = "frb_group3_w16_6"
PRECISION = "ap_fixed<16,6>"
PART = "xcvu9p-flga2104-2L-e"
CLOCK_PERIOD_NS = 5

model = tf.keras.models.load_model("keras_model.keras")
hls_config = hls4ml.utils.config_from_keras_model(
    model, granularity="name", backend="Vitis", default_precision=PRECISION,
)
hls_config["Model"]["Strategy"] = "Resource"
hls_model = hls4ml.converters.convert_from_keras_model(
    model, hls_config=hls_config, output_dir=".", project_name=PROJECT_NAME,
    part=PART, backend="Vitis", io_type="io_stream", clock_period=CLOCK_PERIOD_NS,
)
hls_model.compile()

X_test = np.load("../X_test.npy")
y_test = np.load("../y_test.npy")
y_pred = hls_model.predict(X_test).flatten()
acc = ((y_pred >= 0.5).astype(int) == y_test).mean()
print(f"quantized test accuracy ({PROJECT_NAME}): {acc:.4f}")
