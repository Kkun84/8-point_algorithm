from logging import getLogger
import hydra
import numpy as np
import cv2

import utils


logger = getLogger(__name__)


def main(cfg):
    logger.info('\n' + str(cfg.pretty()))
    images_list = []
    points_list = []
    logger.debug('Read pair images & points.')
    # Load
    for path, points in [[i.path, i.points] for i in cfg.images]:
        logger.info(f"{path=}")
        image = cv2.cvtColor(cv2.imread(hydra.utils.to_absolute_path(path)), cv2.COLOR_BGR2RGB)
        logger.info(f"{image.shape=}")
        images_list.append(image)
        center = np.array(image.shape[:2][::-1]) / 2
        points = np.array(points) - center
        logger.info(f"\n{points=}")
        points_list.append(points)
    assert len(images_list) == 2
    points_list = np.array(points_list)
    assert points_list.shape[0] == 2
    assert points_list.shape[2] == 2
    # Make & save pointed images
    for i, (image, points) in enumerate(zip(images_list, points_list)):
        utils.save_poined_marker_image(image, points, i)
    # Solve F
    if cfg.F is not None:
        F = np.array(cfg.F)
    else:
        F = utils.solve_F(points_list)
    logger.info(f"\n{F=}")
    logger.info(f"{np.linalg.det(F)=}")
    logger.info(f"{np.linalg.det(F[1:, 1:])=}")
    logger.info(f"{np.linalg.det(F[1:, :-1])=}")
    logger.info(f"{np.linalg.det(F[:-1, 1:])=}")
    logger.info(f"{np.linalg.det(F[1:, :-1])=}")
    # Line epipolar
    utils.save_lined_epipolar_image(F.T, images_list[0], points_list[0], points_list[1], 0)
    utils.save_lined_epipolar_image(F, images_list[1], points_list[1], points_list[0], 1)
    return


if __name__ == "__main__":
    hydra.main(config_path='../conf/config.yaml')(main)()
