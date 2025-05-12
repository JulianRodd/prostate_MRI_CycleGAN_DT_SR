import math


def calculate_weighted_metric(metrics):
    fid_score = metrics.get("val_fid_domain", 10.0)
    ssim_score = metrics.get("val_ssim_sr", 0.0)
    psnr_score = metrics.get("val_psnr_sr", 0.0)
    lpips_score = metrics.get("val_lpips_sr", 1.0)
    ncc_score = metrics.get("val_ncc_domain", 0.0)

    norm_fid = 1.0 - math.exp(-0.1 * fid_score) if fid_score < 50 else 1.0
    norm_ssim = 1.0 - ssim_score
    norm_ncc = 1.0 - ncc_score
    norm_psnr = 1.0 - min(max(psnr_score - 20, 0) / 20.0, 1.0)
    norm_lpips = min(lpips_score, 1.0)

    weighted_score = (
        (0.6 * norm_fid)
        + (0.1 * norm_ssim)
        + (0.1 * norm_psnr)
        + (0.1 * norm_lpips)
        + (0.1 * norm_ncc)
    )

    print(f"Weighted metric calculation:")
    print(f"  FID: {fid_score:.4f} (normalized: {norm_fid:.4f}, weight: 0.6)")
    print(f"  SSIM: {ssim_score:.4f} (normalized: {norm_ssim:.4f}, weight: 0.1)")
    print(f"  PSNR: {psnr_score:.4f} (normalized: {norm_psnr:.4f}, weight: 0.1)")
    print(f"  LPIPS: {lpips_score:.4f} (normalized: {norm_lpips:.4f}, weight: 0.1)")
    print(f"  NCC: {ncc_score:.4f} (normalized: {norm_ncc:.4f}, weight: 0.1)")
    print(f"  Weighted score: {weighted_score:.4f}")

    return weighted_score
