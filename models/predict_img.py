import os
import sys
import argparse
import json
from typing import List, Tuple, Optional, Any

import numpy as np
import cv2 as cv
from keras.models import load_model

from chars import find_chars
from char_classifier import CharClassifier, generate_dataset
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------
# IMAGE PREPROCESSING
# ---------------------------
def preprocess_char(char_img: np.ndarray, char_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize, normalize, add channel & batch dimension
    """
    H, W = char_size
    # resize
    char_img = cv.resize(char_img, (W, H), interpolation=cv.INTER_AREA)
    # normalize
    char_img = char_img.astype(np.float32) / 255.0
    # add channel
    char_img = char_img.reshape(H, W, 1)
    # add batch dimension
    char_img = np.expand_dims(char_img, axis=0)
    return char_img


# ---------------------------
# CAPTCHA PREDICTION
# ---------------------------
def predict_from_image(
        image_path: str,
        model_path: str,
        alphabet: str = "0123456789",
        num_chars: int = 4,
        char_size: Tuple[int, int] = (45, 40),
        use_repo_segmentation: bool = True,
        verbose: bool = False
) -> []:
    """
    Predict captcha text from a single image.
    Returns: predicted_text, probabilities (N x C)
    """
    if verbose:
        print(f"Loading model: {model_path}")
    model = CharClassifier()
    model.build((1, 45, 40, 1))  # batch, H, W, C
    model.load_weights()

    # Load image & grayscale
    print(image_path)
    x = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2GRAY)

    h, w = x.shape
    H, W = (45, 80)

    # Reduce image size of it has greater shape than what we want
    if h > H or w > W:
        x = cv.resize(x, (45, 80), interpolation=cv.INTER_AREA)
        h, w = (45, 80)

    # Add borders with the background color to fill the gaps
    diffh, diffw = H - h, W - w

    if diffh > 0 or diffw > 0:
        top, left = diffh // 2, diffw // 2
        bottom, right = top, left

        if diffh % 2 > 0:
            top += 1
        if diffw % 2 > 0:
            left += 1

        x = cv.copyMakeBorder(x, top, bottom, left, right, cv.BORDER_REPLICATE, value=(255, 255, 255))

    # Normalize image pixel intensities in the range [0, 1]
    x = x.astype(np.float32) / 255


    x = find_chars(x, char_size=char_size, num_chars=num_chars)

    x = np.expand_dims(x, axis=-1)


    # Predict
    probs = model.predict(x, batch_size=1)
    y_labels_pred = np.argmax(probs, axis=1)

    num_chars_per_captcha = 4
    alphabet = list("0123456789")

    # 将索引映射为字符
    chars = [alphabet[idx] for idx in y_labels_pred]

    # 每 4 个字符组成一个验证码字符串
    captchas = [''.join(chars[i:i + num_chars_per_captcha])
                for i in range(0, len(chars), num_chars_per_captcha)]

    return captchas


# ---------------------------
# DEBUG SAVE
# ---------------------------
def save_debug(frames_uint8: List[np.ndarray], orig_image_path: str, out_dir: str, probs: np.ndarray,
               predicted_text: str):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(orig_image_path))[0]
    # save original
    img_bgr = cv.imread(orig_image_path)
    cv.imwrite(os.path.join(out_dir, f"{base}_orig.png"), img_bgr)
    # save frames and probs
    for i, f in enumerate(frames_uint8):
        cv.imwrite(os.path.join(out_dir, f"{base}_frame_{i:02d}.png"), f)
    # save metadata
    meta = {
        "predicted_text": predicted_text,
        "per_frame_probs_shape": probs.shape,
        "per_frame_topk": []
    }
    for i in range(probs.shape[0]):
        topk_idx = probs[i].argsort()[::-1][:3].tolist()
        meta["per_frame_topk"].append({"frame": i, "topk_idx": topk_idx, "topk_probs": probs[i, topk_idx].tolist()})
    with open(os.path.join(out_dir, f"{base}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------
# MAIN
# ---------------------------
def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Predict captcha characters from a single image")
    parser.add_argument("--image", "-i", required=True, help="Path to captcha image")
    parser.add_argument("--model", "-m", required=True, help="Path to Keras char model (.h5)")
    parser.add_argument("--alphabet", "-a", default="0123456789", help="Alphabet order in model")
    parser.add_argument("--num-chars", "-n", type=int, default=4, help="Number of characters in captcha")
    parser.add_argument("--char-size", type=int, nargs=2, default=(45, 40), help="Char frame size H W")
    parser.add_argument("--no-repo-seg", dest="use_repo_seg", action="store_false", help="Disable repo segmentation")
    parser.add_argument("--debug-dir", default="predictions_debug", help="Save debug images")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args(argv)

    try:
        predicted_text = predict_from_image(
            image_path=args.image,
            model_path=args.model,
            alphabet=args.alphabet,
            num_chars=args.num_chars,
            char_size=tuple(args.char_size),
            use_repo_segmentation=args.use_repo_seg,
            verbose=args.verbose
        )
        print("Prediction:", predicted_text)
    except ZeroDivisionError as e:
        print("Failed to predict captcha:", e)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
