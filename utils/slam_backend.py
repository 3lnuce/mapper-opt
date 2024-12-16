import random
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.gaussian_renderer import fast_render
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

import numpy as np

from utils.networking import Networking

'''
    Macros
'''

STREAMING = 1

LOG_LOSS = 1
LOG_ERROR = 0
LOG_ERROR_INIT = 0

LOG_VIDEO = 0

LOG_TILE = 1

LOG_TIMING = 0
PRINT_TIMING = 0

tot_forward_v = []
tot_bckward_v = []

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

        self.grad_mask_global = torch.zeros(0, device="cuda").bool()

        if (STREAMING): self.sender = Networking()


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

    def gradual_increase(self, x):
        """
        Function to compute y for a given x, C, M, k, and p.
        
        Parameters:
        x (float or array): The input value(s), where x >= 0.
        C (float): The starting value of y when x is 0.
        M (float): The maximum value of y when x is very large.
        k (float): Scaling parameter to control the curve.
        p (float): Power parameter to control the steepness of the curve.
        
        Returns:
        float or array: The value(s) of y, where C <= y <= M.
        """

        # Parameters
        C = 3    # Starting value of y
        M = 80   # Maximum value of y
        k = 2000 # Controls the scaling of x
        p = 2.0  # Controls the steepness of the curve
        return C + (M - C) * (1 - 1 / (1 + (x / k) ** p))



    def save_image(self, render_pkg, viewpoint, path, ref_pkg=None, is_init=False, iter_idx=None, frame_idx=None, cam_idx=None):
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
        if ref_pkg is not None:
            ref = (
                (torch.clamp(ref_pkg["render"], min=0, max=1.0) * 255)
                .byte()
                .permute(1 ,2 ,0)
                .contiguous()
                .cpu()
                .numpy()
            )
            img_ref = o3d.geometry.Image(ref)
            o3d.io.write_image("%s/%s_render_ref.png" %(path, file_prefix), img_ref)

            # ''' ref error_map '''
            # mask out skipped tiles for computing loss
            # test_rgb_pixel_mask = (image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
            l1_rgb = torch.abs(ref_pkg["render"] - viewpoint.original_image)

            error_map = l1_rgb.detach().mean(dim=0).cpu().numpy()
            cmap = plt.get_cmap("jet")
            error_map = cmap(error_map)
            matplotlib.image.imsave('%s/%s_error_ref.png' %(path, file_prefix), error_map)

            # '''
            # l1_rgb_raw = torch.abs(ref_pkg["render"] - viewpoint.original_image)
            # l1_depth_raw = torch.abs(ref_pkg["depth"] - torch.from_numpy(viewpoint.depth).to(dtype=torch.float32, device=ref_pkg["render"].device)[None])
            # alpha = self.config["Training"]["alpha"] if "alpha" in self.config["Training"] else 0.95


            # torch.save(l1_rgb_raw, "l1_rgb.pt")
            # torch.save(l1_depth_raw, "l1_depth.pt")
            # torch.save(alpha, "alpha.pt")


            # # print (l1_rgb.shape, l1_depth.shape)
            # for row in range(0, 680, 16):
            #     for col in range(0, 1200, 16):
            #         tile_color = l1_rgb_raw[0, row:row+16, col:col+16]
            #         tile_depth = l1_depth_raw[0, row:row+16, col:col+16]
            #         print ('row, col, tile error: ', row, col, alpha * tile_color.mean() + (1 - alpha) * tile_depth.mean())

            # # sys.exit()
            # '''



        # compute and save error map
        image = render_pkg["render"]
        depth = render_pkg["depth"]
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
        gt_image = viewpoint.original_image.cuda()
        _, h, w = gt_image.shape
        mask_shape = (1, h, w)
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]

        '''
        rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
        l1_rgb = torch.abs(image_ab * rgb_pixel_mask - gt_image * rgb_pixel_mask)
        '''
        # mask out skipped tiles for computing loss
        test_rgb_pixel_mask = (image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
        l1_rgb = torch.abs(image * test_rgb_pixel_mask - gt_image * test_rgb_pixel_mask)

        error_map = l1_rgb.detach().mean(dim=0).cpu().numpy()
        cmap = plt.get_cmap("jet")
        error_map = cmap(error_map)
        # matplotlib.image.imsave('%s/frame_%d_cam_%d_iter_%d_error.png' %(path, frame_idx, cam_idx, iter_idx), error_map)
        matplotlib.image.imsave('%s/%s_error.png' %(path, file_prefix), error_map)

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None, last_viewport=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map, last_viewport=last_viewport
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
        if (LOG_ERROR_INIT):
            img_dir = os.path.join(self.save_dir, "images", "init_iter_%d" % (self.init_itr_num))
            mkdir_p(img_dir)

        tic_loop = torch.cuda.Event(enable_timing=True)
        toc_loop = torch.cuda.Event(enable_timing=True)

        tot_forward = 0
        tot_bckward = 0

        # print ("before init.: ", self.gaussians._xyz.shape, " flag: ", self.gaussians.is_active.shape)
        for mapping_iteration in range(self.init_itr_num):
            print ("init. iters: ", mapping_iteration, "/", self.init_itr_num, ", gaussians: ", self.gaussians._xyz.shape, ", flag: ", self.gaussians.is_active.shape)
            tic_loop.record()

            self.iteration_count += 1

            render_pkg = render(
            # render_pkg = fast_render(
                viewpoint, self.gaussians, self.pipeline_params, self.background, render_info="initialization"
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

            if (LOG_ERROR_INIT):
                render_pkg_ref = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, render_info="log_reference"
                )
                self.save_image(
                    render_pkg, viewpoint, img_dir,
                    ref_pkg=render_pkg_ref,
                    is_init=True,
                    iter_idx=mapping_iteration
                )
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )[0]

            if (LOG_TIMING):
                toc_loop.record()
                torch.cuda.synchronize()
                # print("Backend [Init Mapping]: ", tic_loop.elapsed_time(toc_loop))
                tot_forward += tic_loop.elapsed_time(toc_loop)
                # print("tot_forward: ", tot_forward)

            tic_loop.record()

            loss_init.backward()

            if (LOG_TIMING):
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

        # big_points_vs = self.gaussians.max_radii2D > 100
        # prune_mask = big_points_vs
        # self.gaussians.prune_points(prune_mask)

        # self.gaussians.is_active[:] = 0
        # self.gaussians.is_active.int()

        # scale_mask = torch.zeros(self.gaussians.is_active.shape, device='cuda')
        # for param in self.gaussians.optimizer.param_groups:
        #     if (param['name'] == 'scaling'):
        #         param_tensor = param['params'][0]
        #         row_max_scale, _ = torch.max(param_tensor, dim=1)
        #         scale_mask = row_max_scale > 5.0
        #         # param_tensor.grad[scale_mask] = torch.zeros(param_tensor.shape[-1], device='cuda')
        #         self.gaussians.prune_points(scale_mask)

        if (LOG_TIMING):
            print("[Backend] [tot_forward]: ", tot_forward)
            print("[Backend] [tot_bckward]: ", tot_bckward)

            tot_forward_v.append(tot_forward)
            tot_bckward_v.append(tot_bckward)
        return render_pkg

    def map(self, current_window, prune=False, iters=1, frame_idx=-1):
        # print ("before map.: ", self.gaussians._xyz.shape, " flag: ", self.gaussians.is_active.shape)
        # print ("===================================== frame idx: ", frame_idx)
        if (LOG_ERROR):
            img_dir = os.path.join(self.save_dir, "images", "frame_%d_iter_%d_cam_%d" \
                                                % (frame_idx, iters, len(current_window)))
            mkdir_p(img_dir)

        if (LOG_TILE):
            log_tile_dir = os.path.join(self.save_dir, "log_tile", "frame_%d_iter_%d_cam_%d" \
                                            % (frame_idx, iters, len(current_window)))
            mkdir_p(log_tile_dir)

        if (LOG_LOSS):
            file = os.path.join(self.save_dir, "frame_%d_loss.log" %frame_idx)
            f = open(file, "w")

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

        update_all = False
        render_partial = False
        frame_grads = []
        frame_attrs = []
        grad_mask_global = torch.zeros(self.gaussians._xyz.shape[0], device="cuda").bool()
        # print("iters: ", iters)
        # print("curr_window: ", len(current_window))

        num_full = self.gradual_increase(self.gaussians.num_new_inserted)
        # num_full = 150
        print ("===================================================== num full: ", num_full)
        for iter_idx in range(iters):
            print ("map. iters: ", iter_idx, "/", iters, ", gaussians: ", self.gaussians._xyz.shape, ", flag: ", self.gaussians.is_active.shape)

            '''
                Reset active flag and update all gaussians every 10 frames
                Then progressively reduce the number of gaussians to be updated
            '''

            big_points_vs = self.gaussians.max_radii2D > 100
            prune_mask = big_points_vs
            self.gaussians.prune_points(prune_mask)

            # if (iter_idx % 30 == 0):
            #     self.gaussians.is_active[:] = 1
            #     render_partial = False
            #     update_all = True

            #     big_points_vs = self.gaussians.max_radii2D > 100
            #     prune_mask = big_points_vs
            #     self.gaussians.prune_points(prune_mask)
            # else:
            #     render_partial = True
            #     update_all = False

            # if (torch.sum(self.gaussians.is_active) <= 1000):
            #     continue


            if (iter_idx <= num_full):
                # self.gaussians.is_active[:] = 1
                render_partial = False
                update_all = True
            elif (iter_idx <= num_full + 0):
                print ("Partial training !!!")
                # continue
                render_partial = True
                update_all = False
            elif (iter_idx == (iters - 1)):
                print ("Densifying and pruning !!!")
                self.gaussians.densify_and_prune(
                    self.opt_params.densify_grad_threshold,
                    self.gaussian_th,
                    self.gaussian_extent,
                    self.size_threshold,
                )
            else:
                continue


            tic.record()
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []
            freeze_mask_v = []



            for cam_idx in range(len(current_window)):
                tic_loop.record()

                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)

                if (PRINT_TIMING):
                    toc_loop.record()
                    torch.cuda.synchronize()
                    print("before_render: ", tic_loop.elapsed_time(toc_loop))

                render_info = ""
                # if (LOG_TILE):

                render_info = "%s/frame_%d_cam_%d_iter_%d.log" % (log_tile_dir, frame_idx, cam_idx, iter_idx)

                if (prune):
                    render_info = "prune"
                    render_pkg = render(
                        viewpoint, self.gaussians, self.pipeline_params, self.background, render_info=render_info
                    )
                if (not prune):
                    if (update_all):
                        # render_pkg = fast_render(
                        render_pkg = render(
                            viewpoint, self.gaussians, self.pipeline_params, self.background, render_info=render_info, \
                            #iter_num=iter_idx#, render_partial=render_partial#, mask=self.gaussians.is_active.bool()
                        )
                    else:
                        render_pkg = fast_render(
                            viewpoint, self.gaussians, self.pipeline_params, self.background, render_info=render_info, \
                            #iter_num=iter_idx#, render_partial=render_partial#, mask=self.gaussians.is_active.bool()
                    )

                if (PRINT_TIMING):
                    toc_loop.record()
                    torch.cuda.synchronize()
                    print("after_render: ", tic_loop.elapsed_time(toc_loop))

                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                    # freeze_mask,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                    # render_pkg["freeze_mask"],
                )

                # freeze_mask_v.append(freeze_mask)

                active_pixel_mask = (image.sum(dim=0) > self.config["Training"]["rgb_boundary_threshold"]).view(*depth.shape)
                # loss_mapping += get_loss_mapping(
                #     self.config, image, depth, viewpoint, opacity, active_pixel_mask=active_pixel_mask
                # )[0]

                loss_return = get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity, active_pixel_mask=active_pixel_mask
                )
                loss_mapping += loss_return[0]

                f.write("%s\n" % render_info)
                f.write("%f\n" %(loss_return[0]))
            
            
                # l1_rgb_raw = torch.abs(loss_return[1] - loss_return[2])
                # l1_depth_raw = torch.abs(loss_return[3] - loss_return[4])
                # alpha = loss_return[5]

                # torch.save(l1_rgb_raw, "l1_rgb.pt")
                # torch.save(l1_depth_raw, "l1_depth.pt")
                # torch.save(alpha, "alpha.pt")

                # tile_id = []
                # tile_error = []
                # error_map = np.ones((43, 75))
                # # print (l1_rgb.shape, l1_depth.shape)
                # for idx_row, row in enumerate(range(0, 680, 16)):
                #     for idx_col, col in enumerate(range(0, 1200, 16)):
                #         tile_color = l1_rgb_raw[0, row:row+16, col:col+16]
                #         tile_depth = l1_depth_raw[0, row:row+16, col:col+16]
                #         error = alpha * tile_color.mean() + (1 - alpha) * tile_depth.mean()
                #         # print ('row, col, tile error: ', row, col, error)
                #         tile_id.append(idx_row * 75 + idx_col)
                #         tile_error.append(error)
                #         error_map[idx_row, idx_col] = error

                # # unique list, needed additional length to pass
                # # tile_error = sorted(tile_error)
                # # tile_error_raw = tile_error
                # # tile_error = sorted(tile_error, reverse=True)
                # # threshold_idx = int(len(tile_error) * 0.01)
                # # threshold_val = tile_error[threshold_idx]          
                # # viewpoint.tile_list = torch.tensor([idx for idx, val in enumerate(tile_error_raw) if val >= threshold_val]).int()

                # # 2D flag array, less efficient but easy for now
                # error_flat = error_map.flatten()
                # error_flat = sorted(error_flat, reverse=True)
                # threshold_idx = int(len(error_flat) * 0.1)
                # threshold_val = error_flat[threshold_idx]
                # mask = error_map <= threshold_val
                # error_map_filtered = np.where(mask, 0, error_map)
                # error_map_filtered = np.where(error_map_filtered>0, 1, 0)
                # error_map_filtered = torch.tensor(error_map_filtered.flatten()).int().to('cuda')
                # # print ("sum", error_map_filtered.sum())
                # viewpoint.tile_list = error_map_filtered
                # viewpoint.tile_list = torch.zeros(3225, device="cuda").int()

                if (PRINT_TIMING):
                    toc_loop.record()
                    torch.cuda.synchronize()
                    print("get_loss: ", tic_loop.elapsed_time(toc_loop))

                if (LOG_ERROR and cam_idx == 0):
                    if (prune==False and iters > 1 and frame_idx != -1):
                            render_pkg_ref = render(
                                viewpoint, self.gaussians, self.pipeline_params, self.background, render_info="log_reference"
                            )
                            self.save_image(
                                render_pkg, viewpoint, img_dir,
                                ref_pkg=render_pkg_ref,
                                is_init=False,
                                iter_idx=iter_idx,
                                frame_idx=frame_idx,
                                cam_idx=cam_idx
                            )

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

                if (LOG_TIMING):
                    toc_loop.record()
                    torch.cuda.synchronize()
                    # print("Backend [Mapping] cam_idx_wind: ", cam_idx, ", time: ", tic_loop.elapsed_time(toc_loop))
                    tot_forward += tic_loop.elapsed_time(toc_loop)
                    # print("forward_per_iter_seq: ", tic_loop.elapsed_time(toc_loop))

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                tic_loop.record()

                viewpoint = random_viewpoint_stack[cam_idx]

                if (PRINT_TIMING):
                    toc_loop.record()
                    torch.cuda.synchronize()
                    print("before_render: ", tic_loop.elapsed_time(toc_loop))

                render_info = "%s/frame_%d_cam_rand_%d_iter_%d.log" % (log_tile_dir, frame_idx, cam_idx, iter_idx)

                if (prune):
                    render_info = "prune"
                    render_pkg = render(
                        viewpoint, self.gaussians, self.pipeline_params, self.background, render_info=render_info
                    )
                if (not prune):
                    if (update_all):
                        # render_pkg = fast_render(
                        render_pkg = render(
                            viewpoint, self.gaussians, self.pipeline_params, self.background, render_info=render_info#, render_partial=render_partial
                        )
                    else:
                        render_pkg = fast_render(
                            viewpoint, self.gaussians, self.pipeline_params, self.background, render_info=render_info#, render_partial=render_partial
                        )

                if (PRINT_TIMING):
                    toc_loop.record()
                    torch.cuda.synchronize()
                    print("after_render: ", tic_loop.elapsed_time(toc_loop))

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
                active_pixel_mask = (image.sum(dim=0) > self.config["Training"]["rgb_boundary_threshold"]).view(*depth.shape)
                loss_return = get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity, active_pixel_mask=active_pixel_mask
                )
                loss_mapping += loss_return[0]

                f.write("%s\n" % render_info)
                f.write("%f\n" %(loss_return[0]))

                if (PRINT_TIMING):
                    toc_loop.record()
                    torch.cuda.synchronize()
                    print("get_loss: ", tic_loop.elapsed_time(toc_loop))

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

                if (LOG_TIMING):
                    toc_loop.record()
                    torch.cuda.synchronize()
                    # print("Backend [Mapping] cam_idx_rand: ", cam_idx, ", time: ", tic_loop.elapsed_time(toc_loop))
                    tot_forward += tic_loop.elapsed_time(toc_loop)
                    # print("forward_per_iter_rand: ", tic_loop.elapsed_time(toc_loop))

            # print("Backend [Mapping] Iter: ", _, ", time forward: ", tot_forward)

            tic.record()

            scaling = self.gaussians.get_scaling
            ## This slows down training convergence
            # scaling = scaling[self.gaussians.is_active.cpu().bool().numpy()]
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()# + 10 * scaling.mean()

            if (LOG_LOSS):
                formatted_loss = "{:.10f}".format(loss_mapping.item())
                # print ("=================== iter, ", iter_idx, " , loss, ", formatted_loss)
                f.write("iter, %d, loss, %s\n" %(iter_idx, formatted_loss))
            loss_mapping.backward()

            if (LOG_TIMING):
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
                        # # Prune long and thin gaussians
                        # row_max_scale, _ = torch.max(param_tensor, dim=1)
                        # scale_mask = row_max_scale > -8.0
                        # print ("++=+ size of scale mask: ", torch.sum(scale_mask))
                        # self.gaussians.prune_points(scale_mask.cuda())

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
                update_gaussian = False
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True
                    update_all = True

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True
                    update_all = True

                local_grads = []
                local_attrs = []
                
                print ("==== active gaussian num: ", torch.sum(self.gaussians.is_active))

                total_grads = torch.zeros(self.gaussians._xyz.shape[0], device="cuda", dtype=torch.float32)
                if not prune:
                    # True: update all params but select active gaussians for the next round
                    # False: update only active gaussians
                    if update_all == True:
                        for param in self.gaussians.optimizer.param_groups:
                            param_tensor = param['params'][0]
                            if (param_tensor.grad is not None):

                                # param names: xyz, f_dc, f_rest, opacity, scaling, rotation
                                # if ((param['name'] != 'xyz') and (param['name'] != 'scaling')):
                                    # continue
                                if ((param['name'] != 'scaling')):
                                    continue

                                # handle the case for f_rest when not in use
                                if (0 in param_tensor.grad.shape):
                                    continue
                                # print ("=== raw grads: ", param_tensor.grad)
                                # handle the case for f_dc and f_rest when tensor has one more dim
                                if (param['name'] == 'f_dc' or param['name'] == 'f_rest'):
                                    temp_grads = torch.sum(param_tensor.grad, dim=2)
                                    temp_grads = torch.squeeze(temp_grads)
                                    # print ("debug: ", temp_grads.shape)
                                else:
                                    temp_grads = torch.sum(param_tensor.grad, dim=1)
                                # print ("=== param name: ", param['name'])
                                # print ("=== temp_grads: ", temp_grads)
                                # print ("=== total_grads_before: ", total_grads)
                                total_grads += temp_grads
                                print ("=== total_grads_after: ", total_grads)
                        # print ("=== total_grads_final: ", total_grads)

                        max_val = torch.max(total_grads)
                        # print ("=== max_val: ", max_val)
                        sorted_tensor, indices = torch.sort(total_grads.cpu(), descending=True)
                        # plt.plot(range(len(sorted_tensor)), sorted_tensor)
                        # plt.show()
                        # threshold = max_val * 0.2
                        # row_max, _ = torch.max(param_tensor.grad, dim=1)
                        mask = (torch.abs(total_grads) > 0.0001)   # 0.00015 00005 for scale
                        print ("=== mask: ", mask, torch.sum(mask))
                        # self.gaussians.is_active = torch.logical_and(self.gaussians.is_active.bool(), mask).int()
                    else:
                        for param in self.gaussians.optimizer.param_groups:
                            param_tensor = param['params'][0]
                            mask = ~self.gaussians.is_active.cpu().bool().numpy()
                            # print ("=== before: ", param_tensor.grad)
                            param_tensor.grad[mask] = torch.zeros(param_tensor.shape[-1], device='cuda')
                            # print ("=== after: ", param_tensor.grad)



                    # # scale_mask = torch.zeros(self.gaussians.is_active.shape, device='cuda')
                    # # param names: xyz, f_dc, f_rest, opacity, scaling, rotation
                    # for param in self.gaussians.optimizer.param_groups:
                    #     # param_tensor = param['params'][0]
                    #     # print ("=== param name: ", param['name'])
                    #     # print ("[DEBUG_LOG] param grad: ", param_tensor.grad[mask])
                    #     # print ("[DEBUG_LOG] param grad sum: ", param_tensor.grad[mask].sum())

                    #     ## Scaling needs manual reset due to the usage in isotropic_loss
                    #     ## Other params are already zeroed out in backward tile skipping
                    #     if (1):
                    #         param_tensor = param['params'][0]
                    #         if (param_tensor.grad is not None):
                    #             print (param['name'])
                    #             # local_grads.append(param_tensor.grad)
                    #             # print (param_tensor)
                    #             # local_attrs.append(param_tensor.detach().cpu())

                    #             mask = ~self.gaussians.is_active.cpu().bool().numpy()
                    #             # print ("after mask: ", torch.sum(self.gaussians.freeze_mask))
                    #             # mask = ~self.gaussians.freeze_mask.cpu().bool().numpy()
                    #             # print ("mask before clear grads: ", torch.sum(mask))
                    #             # mask = ~mask.cpu().bool().numpy()
                    #             # mask = ~mask
                    #             # print ("mask before clear grads: ", torch.sum(mask))

                    #             # print ("grad before: ", param_tensor.grad)
                    #             param_tensor.grad[mask] = torch.zeros(param_tensor.shape[-1], device='cuda')
                    #             # print ("grad after: ", param_tensor.grad)



                    #             # print ("shape: ", param_tensor.grad.shape)
                    #             if (0 in param_tensor.grad.shape):
                    #                 continue

                    #             '''
                    #             if (param['name'] == 'scaling'):
                    #                 # print ('hit !!!\n');
                    #                 # print (param_tensor)
                    #                 print ("max val: ", torch.max(param_tensor))
                    #                 print ("min val: ", torch.min(param_tensor))
                    #                 print ("max grads: ", torch.max(param_tensor.grad))
                    #                 print ("min grads: ", torch.min(param_tensor.grad))
                    #                 row_max_scale, _ = torch.max(param_tensor, dim=1)
                    #                 scale_mask = row_max_scale > -4.0
                    #                 param_tensor.grad[scale_mask] = torch.zeros(param_tensor.shape[-1], device='cuda')
                    #                 # row_max_scale, _ = torch.max(param_tensor.grad, dim=1)
                    #                 # scale_mask = row_max_scale > -4.0
                    #                 # param_tensor.grad[scale_mask] = torch.zeros(param_tensor.shape[-1], device='cuda')
                    #             '''
                                

                    #             if (param['name'] == 'f_dc' or param['name'] == 'f_rest'):
                    #                 row_max, _ = torch.max(param_tensor.grad, dim=2)
                    #                 row_max = torch.squeeze(row_max, dim=1)
                    #                 # print (row_max, row_max.shape)
                    #             else:
                    #                 row_max, _ = torch.max(param_tensor.grad, dim=1)
                    #                 # print (row_max, row_max.shape)

                    #             max_val = torch.max(param_tensor.grad)
                    #             threshold = max_val * 0.2
                    #             # row_max, _ = torch.max(param_tensor.grad, dim=1)
                    #             grad_mask_local = row_max < threshold
                    #             # print ("local", grad_mask_local, grad_mask_local.shape, torch.sum(grad_mask_local))
                    #             # print ("global", grad_mask_global, grad_mask_global.shape, torch.sum(grad_mask_global))

                    #             self.gaussians.grads_mask = torch.logical_and(self.gaussians.grads_mask, grad_mask_local)
                    # # print ("points removed before scale mask: ", torch.sum(grad_mask_global))
                    # # grad_mask_global = torch.logical_and(grad_mask_global, ~scale_mask)
                    # # print ("points removed after scale mask: ", torch.sum(grad_mask_global))




                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)





                # big_points_vs = self.gaussians.max_radii2D > 100
                # big_points_ws = self.gaussians.get_scaling.max(dim=1).values > 0.1 * self.gaussian_extent
                        
                # prune_mask = torch.logical_or(big_points_vs, big_points_ws)
                # prune_mask = big_points_vs
                # self.gaussians.prune_points(prune_mask)
                # grad_mask_global = grad_mask_global[~prune_mask]



                # Pose update
                # for cam_idx in range(min(frames_to_optimize, len(current_window))):
                #     viewpoint = viewpoint_stack[cam_idx]
                #     if viewpoint.uid == 0:
                #         continue
                #     update_pose(viewpoint)

            # frame_grads.append(local_grads)
            # frame_attrs.append(local_attrs)
            # print ('len grads: ', len(frame_grads))
        print ("===== mask size after each keyframe: ", self.gaussians.grads_mask, self.gaussians.grads_mask.shape, torch.sum(self.gaussians.grads_mask))
        # self.gaussians.is_active = torch.logical_or(self.gaussians.is_active.bool(), grad_mask_global).int()

        # torch.save(frame_grads, "grads.pt")
        # torch.save(frame_attrs, "attrs.pt")




        if LOG_VIDEO and len(viewpoint_stack):
            img_dir = os.path.join(self.save_dir, "video_frames")
            mkdir_p(img_dir)

            render_info = "video"
            viewpoint = viewpoint_stack[0]
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background, render_info=render_info
            )
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
            # img_gt = o3d.geometry.Image(gt)
            # o3d.io.write_image("%s/%s_gt.png" %(path, file_prefix), img_gt)
            o3d.io.write_image("%s/frame_%d.png" %(img_dir, frame_idx), img_render)

        if (LOG_LOSS):
            f.close()
        if (LOG_TIMING):
            print("[Backend] [tot_forward]: ", tot_forward)
            print("[Backend] [tot_bckward]: ", tot_bckward)
            tot_forward_v.append(tot_forward)
            tot_bckward_v.append(tot_bckward)

            print("[Backend] [tot_forward_sum]: ", sum(tot_forward_v))
            print("[Backend] [tot_bckward_sum]: ", sum(tot_bckward_v))

        # return gaussian_split
        # return grad_mask_global

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
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background, render_info="refinement"
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
                # time.sleep(0.01)


                if (PRINT_TIMING):
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

                    if (PRINT_TIMING):
                        toc.record()
                        torch.cuda.synchronize()
                        print("[Backend] [Duration]: ", tic.elapsed_time(toc), " init")

                elif data[0] == "keyframe":
                    # print("cur_frame_idx_back: ", data[1], "keyframe")
                    # print("cur_wind_len: ", len(self.current_window))

                    tic.record()

                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window

                    # python >= 3.7 has keys sorted by default
                    last_vp_key = list(sorted(self.viewpoints.keys()))[-2]
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map, last_viewport=self.viewpoints[last_vp_key])

                    if (PRINT_TIMING):
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

                    if (PRINT_TIMING):
                        toc.record()
                        torch.cuda.synchronize()
                        print("[Backend] [Init KF Opts]: ", tic.elapsed_time(toc))



                    # print ("before active num: ", torch.sum(self.gaussians.is_active))
                    # if (not (0 in self.grad_mask_global.shape)):
                    #     # self.gaussians.is_active = torch.logical_or(self.gaussians.is_active.bool(), self.grad_mask_global).int()
                    #     print (self.grad_mask_global.shape, self.grad_mask_global.shape[0])
                    #     self.gaussians.is_active[:self.grad_mask_global.shape[0]] = self.grad_mask_global

                    # print ("after active num: ", torch.sum(self.gaussians.is_active))

                    # self.gaussians.is_active = torch.logical_and(self.gaussians.is_active.bool(), ~self.gaussians.grads_mask.bool()).int()
                    self.map(self.current_window, iters=iter_per_kf, frame_idx=cur_frame_idx)
                    # self.map(self.current_window[0:1], iters=iter_per_kf, frame_idx=cur_frame_idx)

                    if (STREAMING):
                        data_to_sent = self.gaussians.save_ply("", is_streaming=True)
                        self.sender.send(should_send=True, tensors=data_to_sent)

                    if (PRINT_TIMING):
                        toc.record()
                        torch.cuda.synchronize()
                        print("[Backend] [Mapping]: ", tic.elapsed_time(toc))

                    self.map(self.current_window, prune=True)

                    if (PRINT_TIMING):
                        toc.record()
                        torch.cuda.synchronize()
                        print("[Backend] [Pruning]: ", tic.elapsed_time(toc))

                    self.push_to_frontend("keyframe")

                    if (PRINT_TIMING):
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
