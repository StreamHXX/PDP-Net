from imgaug import augmenters as iaa
from imgaug.augmenters.meta import Augmenter
import numpy as np
import os
import cv2
import random
import shutil
np.bool = np.bool_

class ColumnFPN(Augmenter):
    """
    Simulate Column Fixed Pattern Noise / column gain mismatch.

    Applies a per-column multiplicative gain (and optional additive bias) to create
    vertical banding that follows scene brightness.

    Parameters
    ----------
    strength : tuple(float, float)
        Range for gain variation amplitude. Typical 0.02~0.15.
        This controls how strong the column-to-column gain mismatch is.

    period : tuple(int, int) or None
        Approximate stripe period in pixels. If None, stripes are generated as random
        low-frequency per-column noise (no explicit equal spacing).
        If provided, uses a sinusoidal periodic pattern with slight jitter.

    jitter : float
        Phase jitter strength for the sinusoidal stripes. 0 means perfectly periodic.

    bias_strength : tuple(float, float)
        Optional additive per-column bias (in normalized [0,1] domain). Typical 0~0.02.
        Set to (0.0, 0.0) to disable bias.

    smooth : int
        Smoothing window (odd int) applied to the per-column gain to avoid harsh edges.
        1 disables smoothing.

    per_batch : bool
        If True, uses the same column pattern for all images in a batch (fixed位置更像FPN).
        If False, regenerates pattern per image.

    seed : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        strength=(0.03, 0.12),
        period=(16, 64),
        jitter=0.15,
        bias_strength=(0.0, 0.01),
        smooth=9,
        per_batch=True,
        seed=None,
        name=None,
        deterministic=False,
        random_state=None
    ):
        super().__init__(
            name=name,
            deterministic=deterministic,
            random_state=random_state
        )
        self.strength = strength
        self.period = period
        self.jitter = jitter
        self.bias_strength = bias_strength
        self.smooth = smooth
        self.per_batch = per_batch
        self.seed = seed

    def get_parameters(self):
        return [self.strength, self.period, self.jitter, self.bias_strength,
                self.smooth, self.per_batch, self.seed]

    @staticmethod
    def _moving_average_1d(x, win):
        if win <= 1:
            return x
        win = int(win)
        if win % 2 == 0:
            win += 1
        pad = win // 2
        xp = np.pad(x, (pad, pad), mode="reflect")
        kernel = np.ones(win, dtype=np.float32) / win
        return np.convolve(xp, kernel, mode="valid").astype(np.float32)

    def _make_column_pattern(self, width, rng):
        # gain amplitude
        amp = rng.uniform(self.strength[0], self.strength[1])

        # multiplicative gain per column (centered at 1.0)
        if self.period is not None:
            p = rng.uniform(self.period[0], self.period[1])
            phase = rng.uniform(0, 2*np.pi)
            # small per-column phase jitter -> still "almost equal spacing"
            jitter = rng.normal(0, self.jitter, size=width).astype(np.float32)
            x = (np.arange(width, dtype=np.float32) / p) * (2*np.pi) + phase + jitter
            gain = 1.0 + amp * np.sin(x)
        else:
            # low-frequency random per-column noise (no explicit equal spacing)
            noise = rng.normal(0, 1.0, size=width).astype(np.float32)
            noise = self._moving_average_1d(noise, max(9, self.smooth))
            noise = noise / (np.max(np.abs(noise)) + 1e-6)
            gain = 1.0 + amp * noise

        # optional additive bias per column (normalized)
        bamp = rng.uniform(self.bias_strength[0], self.bias_strength[1])
        if bamp > 0:
            bias = rng.normal(0, 1.0, size=width).astype(np.float32)
            bias = self._moving_average_1d(bias, max(9, self.smooth))
            bias = bias / (np.max(np.abs(bias)) + 1e-6)
            bias = bamp * bias
        else:
            bias = np.zeros((width,), dtype=np.float32)

        # smooth gain to avoid unnatural sharp steps
        gain = self._moving_average_1d(gain.astype(np.float32), self.smooth)
        bias = self._moving_average_1d(bias.astype(np.float32), self.smooth)

        return gain.astype(np.float32), bias.astype(np.float32)

    def _augment_images(self, images, random_state, parents, hooks):
        # imgaug passes a RandomState; we can derive our own rng for reproducibility
        base_rng = np.random.default_rng(self.seed) if self.seed is not None else \
                   np.random.default_rng(random_state.randint(0, 2**31 - 1))

        # If per_batch, create one pattern and reuse
        batch_gain_bias = None

        result = []
        for idx, img in enumerate(images):
            if img is None:
                result.append(img)
                continue

            h, w = img.shape[0], img.shape[1]

            if self.per_batch:
                if batch_gain_bias is None or batch_gain_bias[0].shape[0] != w:
                    gain, bias = self._make_column_pattern(w, base_rng)
                    batch_gain_bias = (gain, bias)
                gain, bias = batch_gain_bias
            else:
                # regenerate per image (still "fixed per image", but not per batch)
                # vary rng slightly per image
                rng = np.random.default_rng(base_rng.integers(0, 2**31 - 1))
                gain, bias = self._make_column_pattern(w, rng)

            # apply: multiply by per-column gain, add per-column bias
            # operate in float32
            img_f = img.astype(np.float32)

            # normalize if uint8/uint16 to [0,1] for stable bias
            if img.dtype == np.uint8:
                scale = 255.0
            elif img.dtype == np.uint16:
                scale = 65535.0
            else:
                scale = None

            if scale is not None:
                img_n = img_f / scale
            else:
                # assume already [0,1] float or similar
                img_n = img_f

            # broadcast gain/bias over rows and channels
            # shape: (1, w, 1) -> (h, w, c)
            gain_ = gain.reshape(1, w, 1)
            bias_ = bias.reshape(1, w, 1)

            img_n = img_n * gain_ + bias_

            # clip and cast back
            img_n = np.clip(img_n, 0.0, 1.0)

            if scale is not None:
                out = (img_n * scale + 0.5).astype(img.dtype)
            else:
                out = img_n.astype(img.dtype)

            result.append(out)

        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

def create_iaa_overexposure():
    a = iaa.Sequential([iaa.pillike.EnhanceBrightness((2.0, 2.5)),
                        iaa.CLAHE(clip_limit=15),
                        iaa.GaussianBlur(sigma=(3.0, 5.0)),
                        iaa.AdditiveGaussianNoise(scale=20),
                        iaa.GammaContrast((0.8, 1.2))])
    return a

def create_iaa_lowlight():

    a = iaa.Sequential([iaa.Multiply(mul=(0.1, 0.35)),
                        iaa.Sequential([iaa.AdditiveGaussianNoise(loc=0, scale=(10, 40)),
                                        iaa.ImpulseNoise(p=(0.01, 0.1)),
                                        iaa.AdditivePoissonNoise(lam=(5, 20))], random_order=True),
                        iaa.OneOf([iaa.WithHueAndSaturation(iaa.WithChannels(0, iaa.Add((-20, 20)))),
                                   iaa.WithBrightnessChannels(iaa.Multiply((0.8, 1.2), per_channel=0.3))]),
                        iaa.Sequential([iaa.GammaContrast(gamma=(1.8, 3.0)),
                                        iaa.OneOf([iaa.GaussianBlur(sigma=(0.8, 1.8)), iaa.MedianBlur(k=(3, 5))]),
                                        iaa.JpegCompression(compression=(70, 90))], random_order=True),
                        iaa.Sequential([iaa.BlendAlphaFrequencyNoise(foreground=iaa.Multiply((0.3, 0.8)),
                                                                     exponent=(-3, 3),
                                                                     upscale_method="linear",
                                                                     size_px_max=64,
                                                                     sigmoid_thresh=10)])], random_order=False)
    return a

def create_motion_blur():
    a = iaa.Sequential([iaa.MotionBlur(k=(10, 20), angle=(-30, 30)),
                        iaa.Sometimes(0.2, iaa.MotionBlur(k=(3, 7))),
                        iaa.LinearContrast(alpha=(0.8, 1.2))])
    return a

HUE_CENTER = {
    "red": 0,
    "orange": 20,
    "yellow": 42,
    "green": 85,
    "cyan": 128,
    "blue": 170,
    "purple": 212,
}

def _keep_one_hue_dominant(images, target_h, band=18,
                           keep_s_mul=(1.0, 1.6),
                           drop_s_mul=(0.0, 0.25)):
    out = []
    for hsv in images:
        hsv2 = hsv.copy()
        H = hsv2[..., 0].astype(np.int16)
        S = hsv2[..., 1].astype(np.float32)

        # circular hue distance
        d = np.minimum((H - target_h) % 256, (target_h - H) % 256)
        mask = d <= band

        ks = np.random.uniform(*keep_s_mul)
        ds = np.random.uniform(*drop_s_mul)

        S[mask]  = np.clip(S[mask]  * ks, 0, 255)
        S[~mask] = np.clip(S[~mask] * ds, 0, 255)

        hsv2[..., 1] = S.astype(np.uint8)
        out.append(hsv2)

    return np.array(out, dtype=np.uint8)

def make_hue_shift_single_dominant(
    dominant_color: str,
    hue_shift=(-50, 50),
    sat_shift=(-50, 50),
    band=18,
    p=1.0
):
    dominant_color = dominant_color.lower()

    if dominant_color == "gray":
        return iaa.Sometimes(p, iaa.Grayscale(alpha=(0.6, 1.0)))

    target_h = HUE_CENTER[dominant_color]

    base = []

    # ✅ 关键修正就在这里
    base.append(
        iaa.AddToHueAndSaturation(
            value_hue=hue_shift,
            value_saturation=sat_shift,
            per_channel=False
        )
    )

    base.append(
        iaa.Lambda(
            func_images=lambda imgs, rs, parents, hooks:
                _keep_one_hue_dominant(imgs, target_h=target_h, band=band)
        )
    )

    return iaa.Sometimes(
        p,
        iaa.WithColorspace(
            to_colorspace="HSV",
            from_colorspace="RGB",
            children=iaa.Sequential(base)
        )
    )

# =========================
# 1) 7 色定义（HSV Hue 中心，uint8）
# =========================
HUE_CENTER = {
    "red": 0,
    "orange": 20,
    "yellow": 42,
    "green": 85,
    "cyan": 128,
    "blue": 170,
    "purple": 212,
}

# =========================
# 2) Saturation gating（只让目标色显眼）
# =========================
def _saturation_gating_keep_one_hue(
    images_hsv_uint8,
    target_h,
    band,
    keep_mul,
    drop_mul,
    sat_floor,
):
    out = []
    for hsv in images_hsv_uint8:
        hsv2 = hsv.copy()
        H = hsv2[..., 0].astype(np.int16)
        S = hsv2[..., 1].astype(np.float32)

        # Hue circular distance
        d = np.minimum((H - target_h) % 256, (target_h - H) % 256)
        mask = d <= band

        ks = np.random.uniform(*keep_mul)
        ds = np.random.uniform(*drop_mul)

        S[mask]  *= ks
        S[~mask] *= ds

        # 防止全灰
        S = np.clip(S, sat_floor, 255)
        hsv2[..., 1] = S.astype(np.uint8)

        out.append(hsv2)

    return np.array(out, dtype=np.uint8)

# =========================
# 3) 单一颜色异常 Augmenter
#    （完全基于你认可的 Hue shift 逻辑）
# =========================
def make_hue_shift_single_color(
    target_color,
    hue_shift,
    band,
    keep_mul,
    drop_mul,
    sat_floor,
    p=1.0,
):
    target_h = HUE_CENTER[target_color]

    return iaa.Sometimes(
        p,
        iaa.WithColorspace(
            to_colorspace="HSV",
            from_colorspace="RGB",
            children=iaa.Sequential([
                # (A) 全图 Hue shift（你的原始方法）
                iaa.WithChannels(0, iaa.Add(hue_shift)),
                # (B) Saturation gating（保证只显眼一种颜色）
                iaa.Lambda(
                    func_images=lambda imgs, rs, parents, hooks:
                        _saturation_gating_keep_one_hue(
                            imgs,
                            target_h=target_h,
                            band=band,
                            keep_mul=keep_mul,
                            drop_mul=drop_mul,
                            sat_floor=sat_floor
                        )
                )
            ])
        )
    )

COMMON_CFG = dict(
    hue_shift=(0, 40),        # 产生足够的 Hue 错色
    band=18,                  # 只保留一种颜色显眼
    keep_mul=(1.1, 1.5),      # 目标色更突出
    drop_mul=(0.25, 0.55),    # 其他色压灰，但不为 0
    sat_floor=25,             # 防止全灰图
)
# =========================
# 4) 构建 7 色增强 dict
# =========================
def build_7color_augdict(prefix="HueShift"):
    d = {}
    for color in HUE_CENTER.keys():
        d[f"{prefix}_{color}"] = iaa.Sequential([
            make_hue_shift_single_color(
                target_color=color,
                p=1.0,
                **COMMON_CFG
            )
        ])
    return d


def img_func(images, random_state, parents, hooks):
    for img in images:
        img[::4] = 0
    return images

def img_func_col(images, random_state, parents, hooks):
    for img in images:
        img[:, ::4] = 0
    return images

def keypoint_func(keypoints_on_images, random_state, parents, hooks):
    return keypoints_on_images

def batch_generation(image_path='', save_path='', mode=None,):
    if mode is None:
        print("select a mode ref official document of imgaug")
        return
    if image_path == '' or save_path == '':
        print('input image path')
        return

    aug_dict = {
        'noise_color': iaa.Sequential([iaa.AdditiveGaussianNoise(scale=0.2*255, per_channel=True)]), #gaosi
        'noise_black': iaa.Sequential([iaa.AdditivePoissonNoise(40)]),
        'drop_block':iaa.Sequential([iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.1), per_channel=True)]),
        # 'blend_temp':iaa.Sequential([iaa.BlendAlphaSimplexNoise(iaa.EdgeDetect(1.0))]),
        'blend_3':iaa.Sequential([iaa.BlendAlphaHorizontalLinearGradient(iaa.AddToHue((-100, 100)))]),
        'Saturation_1': iaa.Sequential([iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB", children=iaa.WithChannels(0, iaa.Add((0, 50))))]),
        'Saturation_2': iaa.Sequential([iaa.Sequential([iaa.AddToHueAndSaturation((-50, 50), per_channel=True)])]),
        'Saturation_3': iaa.Sequential([iaa.MultiplyHueAndSaturation(mul_hue=(0.5, 1.5))]),
        'Saturation_4': iaa.Sequential([iaa.AddToHueAndSaturation((-50, 50), per_channel=True)]),
        'lambda': iaa.Sequential([iaa.Lambda(img_func, keypoint_func)]),
        'spatter': iaa.Sequential([iaa.imgcorruptlike.Spatter(severity=1)]),
        'low_light': create_iaa_lowlight(),
        'overexposure': create_iaa_overexposure(),
        'motion_blur': create_motion_blur(),
    }

    addtional_aug_dict = {} #build_7color_augdict()
    for c in ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "gray"]:
        addtional_aug_dict[f"Saturation_dom_{c}"] = iaa.Sequential([
            make_hue_shift_single_dominant(
                c,
                hue_shift=(-50, 50),
                sat_shift=(-50, 50),
                band=16,
                p=1.0
            )
        ])

    ex_aug_dict = {
        'solarize': iaa.Sequential([iaa.Solarize(0.5, threshold=(32, 128))]),  # gaosi
        'equalize': iaa.Sequential([iaa.pillike.Equalize()]),
        'autocontrast': iaa.Sequential([iaa.pillike.Autocontrast((10, 20), per_channel=True)]),
        'enhancecontrast': iaa.Sequential([iaa.pillike.EnhanceContrast((0.5, 2.0))]),
        'lambda': iaa.Sequential([iaa.Lambda(img_func_col, keypoint_func)]),
    }

    re_aug_dict = {
        #'solarize': iaa.Sequential([iaa.Solarize(1, threshold=(32, 128))]),  # gaosi
        'lambda_colfpn': iaa.Sequential([ColumnFPN( strength=(0.04, 0.12), period=(24, 80), jitter=0.10, bias_strength=(0.0, 0.008), smooth=11, per_batch=True,)]),
    }

    random.seed(1)
    #images = random.sample(os.listdir(image_path), 1000)
    images_names = os.listdir(image_path)
    images = [cv2.imread(os.path.join(image_path, image), cv2.IMREAD_UNCHANGED) for image in images_names]

    if mode == 0:
        for name, augmenter in aug_dict.items():
            print('generating augmented images with mode: {} | ..... '.format(name))

            images_aug = augmenter(images=images)
            save_dir = os.path.join(save_path, name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for image, image_name in zip(images_aug, images_names):
                cv2.imwrite(os.path.join(save_dir, '{}'.format(image_name)), image)

    if mode == 1:

        for name, augmenter in addtional_aug_dict.items():
            print('generating augmented images with mode: {} | ..... '.format(name))

            images_aug = augmenter(images=images)
            save_dir = os.path.join(save_path, name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for image, image_name in zip(images_aug, images_names):
                cv2.imwrite(os.path.join(save_dir, '{}'.format(image_name)), image)

    if mode == 2:

        for name, augmenter in ex_aug_dict.items():
            print('generating augmented images with mode: {} | ..... '.format(name))

            images_aug = augmenter(images=images)
            save_dir = os.path.join(save_path, name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for image, image_name in zip(images_aug, images_names):
                cv2.imwrite(os.path.join(save_dir, '{}'.format(image_name)), image)

    if mode == 3:
        for name, augmenter in re_aug_dict.items():
            print('generating augmented images with mode: {} | ..... '.format(name))

            images_aug = augmenter(images=images)
            save_dir = os.path.join(save_path, name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for image, image_name in zip(images_aug, images_names):
                cv2.imwrite(os.path.join(save_dir, '{}'.format(image_name)), image)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    image_path = r'C:\suu\pythonworks\imgaug\sample_results_rate15_multiview'
    save_path = r'C:\suu\pythonworks\imgaug\auged_results_revised'

    multiview_folders = os.listdir(image_path)
    for folder in multiview_folders:
        folder_path = os.path.join(image_path, folder)
        save_folder_path = os.path.join(save_path, folder)
        batch_generation(image_path=folder_path, save_path=save_folder_path, mode=3)

    # image_path = r'C:\suu\pythonworks\imgaug\img_samples\sample_results_rate50'
    # save_path = r'C:\suu\pythonworks\imgaug\generate_img_3'
    # batch_generation(image_path=image_path, save_path=save_path, mode=1)



