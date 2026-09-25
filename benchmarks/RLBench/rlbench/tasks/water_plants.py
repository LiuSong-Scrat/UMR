import os
from typing import List
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.const import PrimitiveShape
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition

WATER_NUM = 5
PLANT_COLLISION_MODE = os.environ.get(
    'RLBENCH_WATER_PLANT_COLLISION', 'enabled').lower()
WATER_DROP_COLLISION_MODE = os.environ.get(
    'RLBENCH_WATER_DROP_COLLISION', 'original').lower()


class WaterPlants(Task):

    def init_task(self) -> None:
        if WATER_DROP_COLLISION_MODE not in {'disabled', 'original'}:
            raise ValueError(
                'RLBENCH_WATER_DROP_COLLISION must be original or disabled, '
                f'got {WATER_DROP_COLLISION_MODE!r}.')
        self.drops = []
        self.plant = Shape('plant')
        self.success_sensor = ProximitySensor('success')
        self.pour_point = ProximitySensor('pour_point')
        self.waterer = Shape('waterer')
        self.head = Shape('head')
        self._configure_plant_collision()
        self.register_graspable_objects([self.waterer])
        self.pour_point_reached = DetectedCondition(
            self.head, self.pour_point)

    def _configure_plant_collision(self) -> None:
        """Keep the decorative plant visible without blocking the robot."""
        if PLANT_COLLISION_MODE in {'enabled', 'on'}:
            return
        if PLANT_COLLISION_MODE not in {
                'disabled', 'off', 'penetrable', 'pass_through'}:
            raise ValueError(
                'RLBENCH_WATER_PLANT_COLLISION must be enabled or disabled, '
                f'got {PLANT_COLLISION_MODE!r}.')

        # Do not traverse the plant tree: a separately named pot/soil child
        # must retain its original collision properties.
        self.plant.set_collidable(False)
        self.plant.set_respondable(False)

    def init_episode(self, index: int) -> List[str]:
        self.register_success_conditions(
            [DetectedCondition(self.waterer, self.success_sensor)])
        self.reached = False
        self.reachedOnce = False
        return ['water plant',
                'pick up the watering can by its handle and water the plant',
                'pour some water on the plant',
                'the plant needs hydration',
                'pour water from the watering can into the plant pot',
                'water the soil']

    def variation_count(self) -> int:
        return 1

    def step(self) -> None:
        if not self.reached:
            self.reached = self.pour_point_reached.condition_met()[0]
            if self.reached and not self.reachedOnce:
                for i in range(WATER_NUM):
                    if WATER_DROP_COLLISION_MODE == 'original':
                        # Reproduce the original RLBench task's default drop
                        # physics instead of forcing sensor-only markers.
                        drop = Shape.create(
                            PrimitiveShape.CUBOID,
                            mass=0.0001,
                            size=[0.005, 0.005, 0.005],
                        )
                    else:
                        # The default evaluation mode treats drops as sensors
                        # only and prevents physical interference.
                        drop = Shape.create(
                            PrimitiveShape.CUBOID,
                            mass=0.0001,
                            size=[0.005, 0.005, 0.005],
                            respondable=False,
                        )
                        drop.set_collidable(False)
                    drop.set_color([0.1, 0.1, 0.9])
                    pos = list(np.random.normal(0, 0.0005, size=(3,)))
                    drop.set_position(pos,
                                      relative_to=self.head)
                    self.drops.append(drop)
                self.register_success_conditions(
                    [DetectedCondition(self.drops[i], self.success_sensor) for i
                     in range(WATER_NUM)])
                self.reachedOnce = True

    def cleanup(self) -> None:
        for d in self.drops:
            d.remove()
        self.drops = []
