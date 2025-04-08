import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_conditional, EMA
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
import gc

gc.collect()
#torch.cuda.empty_cache()
#torch.cuda.ipc_collect()
# logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

logging.basicConfig(
    filename=f'weights/training.log', 
    level=logging.DEBUG,     
    format='%(asctime)s - %(message)s',  
    filemode='a'  
)

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cpu"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    # def sample(self, model, n, labels, cfg_scale=3):
    #     logging.info(f"Sampling {n} new images....")
    #     model.eval()
    #     with torch.no_grad():
    #         x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
    #         for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
    #             t = (torch.ones(n) * i).long().to(self.device)
    #             predicted_noise = model(x, t, labels)

    #             if cfg_scale > 0:
    #                 uncond_predicted_noise = model(x, t, None)
    #                 predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
    #             alpha = self.alpha[t][:, None, None, None]
    #             alpha_hat = self.alpha_hat[t][:, None, None, None]
    #             beta = self.beta[t][:, None, None, None]
    #             if i > 1:
    #                 noise = torch.randn_like(x)
    #             else:
    #                 noise = torch.zeros_like(x)
    #             x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
    #     model.train()
    #     x = (x.clamp(-1, 1) + 1) / 2
    #     x = (x * 255).type(torch.uint8)
    #     return x

    def sample(self, model, n, rain_ffts, cfg_scale=5):
        """
        model : the diffusion model that was trained 
        n : the number of samples, basically the batch size
        rain_ffts : normalized rain_ffts of the noise images, this will be used for conditioning

        return : a (6xWxH) image representing the difference in FFT given the condition
        """

        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 6, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                # predicted_noise = model(x, t, labels)
                x = x.to(self.device)
                rain_ffts = rain_ffts.to(self.device)
                predicted_noise = model(x, t, rain_ffts)

                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        return x

def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet_conditional(args.image_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    stats = {
        "best_train_epoch": -1,
        "best_train_loss" : float("infinity"),
        "avg_train_loss": []
    }
    prev_best_filename = None  
    auto_delete = True 
    dir = "weights"

    if args.load_state_dict:
        weights_path = f"weights/{args.epoch_start}.pth"
        print(f"loading {weights_path}")
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)

    try:
        for epoch in range(args.epochs):
            avg_train_loss = 0
            num_train_items = 0
            actual_epoch = epoch + args.epoch_start
            if args.load_state_dict:
                actual_epoch += 1
            logging.info(f"Starting epoch {actual_epoch}:")
            pbar = tqdm(dataloader)
            for i, (diff_fft, rain_fft) in enumerate(pbar):
                diff_fft = diff_fft.to(device)
                rain_fft = rain_fft.to(device)
                t = diffusion.sample_timesteps(diff_fft.shape[0]).to(device)
                x_t, noise = diffusion.noise_images(diff_fft, t)
                if np.random.random() < 0.1:
                    rain_fft = None
                with autocast():
                    predicted_noise = model(x_t, t, rain_fft)
                    loss = mse(noise, predicted_noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ema.step_ema(ema_model, model)

                pbar.set_postfix(MSE=loss.item())
                logger.add_scalar("MSE", loss.item(), global_step=actual_epoch * l + i)
                avg_train_loss += loss.item() * diff_fft.shape[0]
                num_train_items += diff_fft.shape[0]

            avg_train_loss /= num_train_items
            logging.info(f"MSE: {avg_train_loss}")
            stats["avg_train_loss"].append(float(avg_train_loss))

            if avg_train_loss < stats["best_train_loss"]:
                if  auto_delete and prev_best_filename and os.path.exists(prev_best_filename):
                    os.remove(prev_best_filename)
                    print(f"Deleted previous best model: {prev_best_filename}")
                    
                stats["best_train_loss"] = avg_train_loss
                stats["best_train_epoch"] = actual_epoch

                filename = f'{dir}/{actual_epoch}.pth'
                torch.save(model.state_dict(), filename)
                print(f"Found better training loss {avg_train_loss}, Saved {filename}.")
                prev_best_filename = filename

            if epoch == args.epochs - 1:
                filename = f'{dir}/{actual_epoch}.pth'
                torch.save(model.state_dict(), filename)
                print("Saved {filename} at last epoch.")


    except Exception as e:
        print("Exception happened", e)
    finally:
        title = f"FFT Diffusion model"
        epochs = len(list(stats["avg_train_loss"]))

        plt.figure(figsize=(12, 8))
        plt.plot(range(epochs), list(stats["avg_train_loss"]), label="avg_train_loss", linestyle='-', color='g')
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss/MSE")
        plt.legend()
        # plt.ylim(0, 3)
        plt.savefig(f"{dir}/plot.png")
        filename = f'{dir}/{actual_epoch}.pth'
        print(f"Emergency save! Saving to {filename}")
        torch.save(model.state_dict(), filename)
        logging.info(f"Stats: {stats}")

        # if epoch % 10 == 0:
        #     labels = torch.arange(10).long().to(device)
        #     sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
        #     ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
        #     plot_images(sampled_images)
        #     save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        #     save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
        #     torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
        #     torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
        #     torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_conditional"
    args.epochs = 22
    args.batch_size = 1
    args.image_size = 128
    args.num_classes = None
    args.dataset_path = f'/home/bernerd/eecs556/data.csv'
    args.diff_stats_csv_file = f'statistics/diff_fft_statistics_log.csv'
    args.rain_stats_csv_file = f'statistics/rain_fft_statistics_log.csv'
    args.device = "cpu"
    args.load_state_dict = False
    args.epoch_start = 0
    args.lr = 3e-4
    train(args)


if __name__ == '__main__':
    launch()
    # device = "cuda"
    # model = UNet_conditional(num_classes=10).to(device)
    # ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8
    # y = torch.Tensor([6] * n).long().to(device)
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # plot_images(x)
