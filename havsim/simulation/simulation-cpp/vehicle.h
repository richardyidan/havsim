#ifndef HAVSIM_CPP_VEHICLE_H
#define HAVSIM_CPP_VEHICLE_H

#include <vector>

class Lane;

class Vehicle {
public:
    Vehicle(int vehid,
            Lane *lane,
            std::vector<double> cf_parameters,
            std::vector<double> lc_parameters,
            double length = 2,
            double relaxp = -1
    );

    void initialize(double pos,
                    double speed,
                    double headway,
                    int inittime,
                    Vehicle *lead = nullptr,
                    Vehicle *fol = nullptr,
                    Vehicle *lfol = nullptr,
                    Vehicle *rfol= nullptr,
                    Vehicle *llead= nullptr,
                    Vehicle *rlead= nullptr);

    // car following model: updates the action (acceleration by default)
    virtual void call_cf(int timeind, double dt);

    virtual double IDM(const std::vector<double>& p, const std::vector<double> &state);

    double get_speed() const;
    double get_pos() const;
    double get_len() const;
    int get_vehid() const;
    void update(int timeind, double dt);
    void update_headway();
    std::vector<double> get_posmem() const;
    Vehicle* get_lead() const;

private:
    // id of the vehicle
    int vehid_;
    Lane *lane_;

    // ====== Model parameters =====
    std::vector<double> cf_parameters_;
    std::vector<double> lc_parameters_;
    double relaxp_;
    double length_;

    // Leader/follower relationships
    Vehicle *lead_;
    Vehicle *fol_;
    Vehicle *lfol_;
    Vehicle *rfol_;
    Vehicle *llead_;
    Vehicle *rlead_;

    // State
    double pos_;
    double speed_;
    double headway_;
    double action_;

    // Memory
    int inittime_;
    int endtime_;
    std::vector<std::pair<Vehicle *,int>> leadmem_;
    std::vector<std::pair<Lane *,int>> lanemem_;
    std::vector<double> posmem_;
    std::vector<double> speedmem_;
    std::vector<std::pair<std::pair<int,int>,std::vector<double>>> relaxmem_;

    bool in_relax_;
    std::vector<double> relax_;
    int relax_start_;

};

#endif //HAVSIM_CPP_VEHICLE_H
