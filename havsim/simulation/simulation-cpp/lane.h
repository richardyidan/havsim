
#ifndef HAVSIM_CPP_LANE_H
#define HAVSIM_CPP_LANE_H
enum LaneMethod{
    SPEED,FREE,FLOW,MERGE
};
class Vehicle;
class Lane{
public:
    explicit Lane(LaneMethod lane_method);
    double call_downstream(Vehicle* vehicle, int timeind, double dt);

private:
    double speed_fun(double speed);
    LaneMethod lane_method_;

};
#endif //HAVSIM_CPP_LANE_H
