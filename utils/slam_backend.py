import random
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping

import os
import time
import matplotlib
import matplotlib.pyplot as plt
import open3d as o3d
from gaussian_splatting.utils.system_utils import mkdir_p

'''
    Macros
'''
TIMING = 1
LOG_ERROR = 1

class BackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        self.save_dir = None

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )


    def save_image(self, render_pkg, viewpoint, path, is_init=False, iter_idx=None, frame_idx=None, cam_idx=None):
        if is_init:
            file_prefix = "init_iter_%d" % (iter_idx)
        else:
            file_prefix = "frame_%d_cam_%d_iter_%d" % (frame_idx, cam_idx, iter_idx)

        rgb = (
            (torch.clamp(render_pkg["render"], min=0, max=1.0) * 255)
            .byte()
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy()
        )

        gt = (
            (torch.clamp(viewpoint.original_image, min=0, max=1.0) * 255)
            .byte().
            permute(1 ,2 ,0).
            contiguous()
            .cpu()
            .numpy()
        )

        # save reference and rendered image
        img_render = o3d.geometry.Image(rgb)
        img_gt = o3d.geometry.Image(gt)
        o3d.io.write_image("%s/%s_gt.png" %(path, file_prefix), img_gt)
        o3d.io.write_image("%s/%s_render.png" %(path, file_prefix), img_render)

        # compute and save error map
        image = render_pkg["render"]
        depth = render_pkg["depth"]
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
        gt_image = viewpoint.original_image.cuda()
        _, h, w = gt_image.shape
        mask_shape = (1, h, w)
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
        l1_rgb = torch.abs(image_ab * rgb_pixel_mask - gt_image * rgb_pixel_mask)

        error_map = l1_rgb.detach().mean(dim=0).cpu().numpy()
        cmap = plt.get_cmap("jet")
        error_map = cmap(error_map)
        # matplotlib.image.imsave('%s/frame_%d_cam_%d_iter_%d_error.png' %(path, frame_idx, cam_idx, iter_idx), error_map)
        matplotlib.image.imsave('%s/%s_error.png' %(path, file_prefix), error_map)

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def initialize_map(self, cur_frame_idx, viewpoint, frame_idx=-1):
        if (LOG_ERROR):
            img_dir = os.path.join(self.save_dir, "images", "init_iter_%d" % (self.init_itr_num))
            mkdir_p(img_dir)

        tic_loop = torch.cuda.Event(enable_timing=True)
        toc_loop = torch.cuda.Event(enable_timing=True)

        tot_forward = 0
        tot_bckward = 0

        for mapping_iteration in range(self.init_itr_num):
            tic_loop.record()

            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )

            if (LOG_ERROR):
                self.save_image(
                    render_pkg, viewpoint, img_dir,
                    is_init=True,
                    iter_idx=mapping_iteration
                )

            if (TIMING):
                toc_loop.record()
                torch.cuda.synchronize()
                # print("Backend [Init Mapping]: ", tic_loop.elapsed_time(toc_loop))
                tot_forward += tic_loop.elapsed_time(toc_loop)
                # print("tot_forward: ", tot_forward)

            tic_loop.record()

            loss_init.backward()

            if (TIMING):
                toc_loop.record()
                torch.cuda.synchronize()
                # print("Backend [Init Mapping] Loss: ", tic_loop.elapsed_time(toc_loop))
                tot_bckward += tic_loop.elapsed_time(toc_loop)
                # print("tot_bckward: ", tot_bckward)

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")

        print("[Backend] [tot_forward]: ", tot_forward)
        print("[Backend] [tot_bckward]: ", tot_bckward)
        logger.info('test')

        return render_pkg

    def map(self, current_window, prune=False, iters=1, frame_idx=-1):
        if (LOG_ERROR):
            img_dir = os.path.join(self.save_dir, "images", "frame_%d_iter_%d_cam_%d" \
                                                % (frame_idx, iters, len(current_window)))
            mkdir_p(img_dir)

        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        tic_loop = torch.cuda.Event(enable_timing=True)
        toc_loop = torch.cuda.Event(enable_timing=True)

        tot_forward = 0
        tot_bckward = 0

        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        # print("iters: ", iters)
        # print("curr_window: ", len(current_window))
        for iter_idx in range(iters):
            tic.record()
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []

            for cam_idx in range(len(current_window)):
                tic_loop.record()

                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )

                if (LOG_ERROR):
                    if (prune==False and iters > 1 and frame_idx != -1):
                            self.save_image(
                                render_pkg, viewpoint, img_dir,
                                is_init=False,
                                iter_idx=iter_idx,
                                frame_idx=frame_idx,
                                cam_idx=cam_idx
                            )

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

                if (TIMING):
                    toc_loop.record()
                    torch.cuda.synchronize()
                    # print("Backend [Mapping] cam_idx_wind: ", cam_idx, ", time: ", tic_loop.elapsed_time(toc_loop))
                    tot_forward += tic_loop.elapsed_time(toc_loop)
                    # print("tot_forward: ", tot_forward)

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                tic_loop.record()

                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

                if (TIMING):
                    toc_loop.record()
                    torch.cuda.synchronize()
                    # print("Backend [Mapping] cam_idx_rand: ", cam_idx, ", time: ", tic_loop.elapsed_time(toc_loop))
                    tot_forward += tic_loop.elapsed_time(toc_loop)
                    # print("tot_forward: ", tot_forward)

            # print("Backend [Mapping] Iter: ", _, ", time forward: ", tot_forward)

            tic.record()

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()

            if (TIMING):
                toc.record()
                torch.cuda.synchronize()
                # print("Backend [Mapping] Loss: ", tic.elapsed_time(toc))
                tot_bckward += tic.elapsed_time(toc)
                # print("tot_bckward: ", tot_bckward)

            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)

                # Pose update
                # for cam_idx in range(min(frames_to_optimize, len(current_window))):
                #     viewpoint = viewpoint_stack[cam_idx]
                #     if viewpoint.uid == 0:
                #         continue
                #     update_pose(viewpoint)

        print("[Backend] [tot_forward]: ", tot_forward)
        print("[Backend] [tot_bckward]: ", tot_bckward)

        return gaussian_split

    def color_refinement(self):
        Log("Starting color refinement")

        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            )
            image, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (
                Ll1
            ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
        Log("Map refinement done")

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"

        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)

    def run(self):
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue

                tic.record()

                self.map(self.current_window)
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()

                if (TIMING):
                    toc.record()
                    torch.cuda.synchronize()
                    print("[Backend] [Duration]: ", tic.elapsed_time(toc), " routine")

            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend()

                elif data[0] == "init":
                    print("cur_frame_idx_back: ", data[1], "init")

                    tic.record()

                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    Log("Resetting the system")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    self.initialize_map(cur_frame_idx, viewpoint, frame_idx=cur_frame_idx)
                    self.push_to_frontend("init")

                    if (TIMING):
                        toc.record()
                        torch.cuda.synchronize()
                        print("[Backend] [Duration]: ", tic.elapsed_time(toc), " init")

                elif data[0] == "keyframe":
                    print("cur_frame_idx_back: ", data[1], "keyframe")
                    # print("cur_wind_len: ", len(self.current_window))

                    tic.record()

                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)

                    if (TIMING):
                        toc.record()
                        torch.cuda.synchronize()
                        print("[Backend] [Add KFs]: ", tic.elapsed_time(toc))

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 10
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    for cam_idx in range(len(self.current_window)):
                        if self.current_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize:
                            # print ("cam_rot_delta: ", viewpoint.cam_rot_delta)
                            # print ("lr: ", self.config['Training']['lr']['cam_rot_delta'])
                            # print ("cam_trans_delta: ", viewpoint.cam_trans_delta)
                            # print ("lr: ", self.config['Training']['lr']['cam_trans_delta'])
                            # print ("opt_params len: ", len(opt_params))
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
                                    "name": "trans_{}".format(viewpoint.uid),
                                }
                            )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_a],
                                "lr": 0.01,
                                "name": "exposure_a_{}".format(viewpoint.uid),
                            }
                        )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_b],
                                "lr": 0.01,
                                "name": "exposure_b_{}".format(viewpoint.uid),
                            }
                        )
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)

                    if (TIMING):
                        toc.record()
                        torch.cuda.synchronize()
                        print("[Backend] [Init KF Opts]: ", tic.elapsed_time(toc))

                    self.map(self.current_window, iters=iter_per_kf, frame_idx=cur_frame_idx)

                    if (TIMING):
                        toc.record()
                        torch.cuda.synchronize()
                        print("[Backend] [Mapping]: ", tic.elapsed_time(toc))

                    self.map(self.current_window, prune=True)

                    if (TIMING):
                        toc.record()
                        torch.cuda.synchronize()
                        print("[Backend] [Pruning]: ", tic.elapsed_time(toc))

                    self.push_to_frontend("keyframe")

                    if (TIMING):
                        toc.record()
                        torch.cuda.synchronize()
                        print("[Backend] [Duration]: ", tic.elapsed_time(toc), " keyframe", "\n")

                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return
