#include <fstream>
#include <random>
#include <chrono>

#include "IOUtils.h"

void PreProcessor::read_map_rows(const std::string& map_file, std::vector<std::string>& rows) {
    std::ifstream in(map_file);
    if (!in.is_open()) { std::cerr << "Cannot open " << map_file << "\n"; std::exit(1); }
    std::string line, key; int H=0, W=0;

    std::getline(in, line);        // e.g., "type octile"
    in >> key >> H >> key >> W;    // "height H" "width W"
    std::getline(in, line);        // eol
    std::getline(in, line);        // "map"
    rows.clear(); rows.reserve(H);
    for (int i=0;i<H;i++){
        std::getline(in, line);
        if ((int)line.size()!=W) { std::cerr<<"Bad map row width\n"; std::exit(1); }
        rows.push_back(line);
    }
}

std::vector<std::vector<double>> PreProcessor::load_cost_grids(const std::vector<std::string>& cost_files, int H, int W) {
    std::vector<std::vector<double>> grids;
    grids.reserve(cost_files.size());
    for (const auto& path : cost_files){
        std::ifstream in(path);
        if (!in.is_open()) { std::cerr<<"Cannot open "<<path<<"\n"; std::exit(1); }
        std::vector<double> g; g.reserve(H*W);
        for (int r=0;r<H;r++){
            for (int c=0;c<W;c++){
                long long v=0; in >> v;
                g.push_back(double(v));
            }
        }
        grids.push_back(std::move(g));
    }
    return grids;
}


void split_string(std::string string, std::string delimiter, std::vector<std::string> &results)
{
    size_t first_delimiter;

    while ((first_delimiter = string.find_first_of(delimiter)) != string.npos) {
        if (first_delimiter > 0) {
            results.push_back(string.substr(0, first_delimiter));
        }
        string = string.substr(first_delimiter + 1);
    }

    if (string.length() > 0) {
        results.push_back(string);
    }
}

bool load_gr_files(std::vector<std::string> gr_files, std::vector<Edge> &edges_out, size_t &graph_size){
  size_t          max_node_num = 0;
  for (auto gr_file: gr_files){
    std::ifstream file(gr_file.c_str());
    
    if (file.is_open() == false){
      std::cerr << "cannot open the gr file " << gr_file << std::endl;
      return false;
    }

    std::string line;
    int idx_edge = 0;
    while (file.eof() == false) {
        std::getline(file, line);

        if (line == "") {
            break;
        }

        std::vector<std::string> decomposed_line;
        split_string(line, " ", decomposed_line);

        std::string type = decomposed_line[0];
        if ((std::strcmp(type.c_str(),"c") == 0) || (std::strcmp(type.c_str(),"p") == 0)) {
            continue; //comment or problem lines, not part of the graph
        }

        if (std::strcmp(type.c_str(),"a") == 0) { //arc
          if (idx_edge < (int)edges_out.size() - 1){
            if (
                (stoul(decomposed_line[1]) != edges_out[idx_edge].source) ||
                (stoul(decomposed_line[2]) != edges_out[idx_edge].target)) {
              // arc_sign src dest should be same in both files
              std::cerr << "file inconsistency" << std::endl;
              return false;
            }
            edges_out[idx_edge].cost.push_back(std::stoul(decomposed_line[3]));
          }else{
            Edge e(std::stoul(decomposed_line[1]),
                   std::stoul(decomposed_line[2]),
                   {std::stoul(decomposed_line[3])});
            edges_out.push_back(e);
            max_node_num = std::max({max_node_num, e.source, e.target});
          }
        }
        idx_edge ++;
    }
    file.close();
  }
  graph_size = max_node_num;
  return true;
}

bool load_gr_files(std::string gr_file1, std::string gr_file2, std::vector<Edge> &edges_out, size_t &graph_size) {
    size_t          max_node_num = 0;
    std::ifstream   file1(gr_file1.c_str());
    std::ifstream   file2(gr_file2.c_str());

    if ((file1.is_open() == false) || (file2.is_open() == false)) {
        return false;
    }

    std::string line1, line2;
    while ((file1.eof() == false) && (file2.eof() == false)) {
        std::getline(file1, line1);
        std::getline(file2, line2);

        if ((line1 == "") || (line2 == "")) {
            break;
        }

        std::vector<std::string> decomposed_line1, decomposed_line2;
        split_string(line1, " ", decomposed_line1);
        split_string(line2, " ", decomposed_line2);

        std::string type = decomposed_line1[0];
        if ((std::strcmp(type.c_str(),"c") == 0) || (std::strcmp(type.c_str(),"p") == 0)) {
            continue; //comment or problem lines, not part of the graph
        }

        if ((decomposed_line1[0] != decomposed_line2[0]) ||
            (decomposed_line1[1] != decomposed_line2[1]) ||
            (decomposed_line1[2] != decomposed_line2[2])) {
            // arc_sign src dest should be same in both files
            return false;
        }

        if (std::strcmp(type.c_str(),"a") == 0) { //arc
            Edge e(std::stoul(decomposed_line1[1]),
                   std::stoul(decomposed_line1[2]),
                   {std::stoul(decomposed_line1[3]), std::stoul(decomposed_line2[3])});
            edges_out.push_back(e);
            max_node_num = std::max({max_node_num, e.source, e.target});
        }
    }
    graph_size = max_node_num;
    return true;
}

bool load_txt_file(std::string txt_file, std::vector<Edge> &edges_out, size_t &graph_size) {
    bool            first_line = true;
    size_t          max_node_num = 0;
    std::ifstream   file(txt_file.c_str());

    if (file.is_open() == false) {
        return false;
    }

    std::string line;
    while (file.eof() == false) {
        std::getline(file, line);

        if (line == "") {
            break;
        }

        std::vector<std::string> decomposed_line;
        split_string(line, " ", decomposed_line);

        if (first_line) {
            first_line = false;
            continue;
        }
        Edge e(std::stoul(decomposed_line[0]),
               std::stoul(decomposed_line[1]),
               {std::stoul(decomposed_line[2]), std::stoul(decomposed_line[3])});
        edges_out.push_back(e);
        max_node_num = std::max({max_node_num, e.source, e.target});
    }
    graph_size = max_node_num;
    return true;
}


bool load_queries(std::string query_file, std::vector<std::pair<size_t, size_t>> &queries_out) {
    std::ifstream   file(query_file.c_str());

    if (file.is_open() == false) {
        return false;
    }

    std::string line;
    while (file.eof() == false) {
        std::getline(file, line);

        if (line == "") {
            break;
        } else if (line[0] == '#') {
            continue; // Commented out queries
        }

        std::vector<std::string> decomposed_line;
        split_string(line, ",", decomposed_line);

        std::pair<size_t, size_t> query = {std::stoul(decomposed_line[0]), std::stoul(decomposed_line[1])};
        queries_out.push_back(query);
    }
    return true;
}


void PreProcessor::read_map(std::string map_file_name, Map& map, std::unordered_map<size_t, std::vector<int>>& id2coord)
{
    std::ifstream Input(map_file_name);
    if (!Input.is_open()) {
        std::cout << "Error: Unable to open map file." << std::endl;
        exit(1);
    }
    
    // Read map information
    std::string temp;
    int height, width;
    std::getline(Input, temp);
    Input >> temp >> height >> temp >> width;
    std::getline(Input, temp);
    std::getline(Input, temp);

    if (height <= 0 || width <= 0) {
        std::cout << "Error: Invalid map format." << std::endl;
        exit(1);
    }

    // Create a Map object to store the map data
    map.setSize(width, height);

    // print the width and height
    std::cout << "[IOUtils.cpp l210]Map dimensions: width = " << width << ", height = " << height << std::endl;

    // Read the map data from the file and store it in the Map object
    for (int x = 0; x < height; ++x) {
        std::string line;
        std::getline(Input, line);

        for (int y = 0; y < width; ++y) {
            if (line[y] == '@' || line[y] == 'T') {
                map.setVal(x, y, -1);
                map.setID(x, y, -1);
            } else if (line[y] == '.' or line[y] == 'L' or line[y] == 'F' or line[y] == 'C') { // . free cell, L landmark, F fire, C civilian
                map.setVal(x, y, 0);
                map.setID(x, y, map.graph_size);
                id2coord.insert(std::make_pair(map.graph_size, std::vector<int>({x, y})));
                map.graph_size ++; // Increment graph_size for each '.'
            } else {
                std::cout << x << y << " " << line[y];
                getchar();
                std::cout << "Error: Invalid character in map file." << std::endl;
                exit(1);
            }
        }
    }

    Input.close();
}

void PreProcessor::read_scenario(std::string config_file,
                                 Map& map,
                                 int agent_num,
                                 std::vector<std::pair<size_t, size_t>>& start_end,
                                 int refine_R)
{   
    refine_R = 1;
    auto to_refined = [&](int x, int y, int R){
        if (R <= 1) return std::pair<int,int>(x,y);
        return std::pair<int,int>(x*R + R/2, y*R + R/2); // center subcell
    };

    auto in_bounds = [&](int x, int y){
        return (0 <= x && x < map.width) && (0 <= y && y < map.height);
    };

    auto print_oob = [&](int line_no, const char* tag, int x, int y){
        std::cerr << "[SCEN OOB] line " << line_no
                  << " " << tag << "=(" << x << "," << y
                  << ") out of bounds for map " << map.width << "x" << map.height << "\n";
    };

    start_end.clear();

    // ------------------------ Case 1: simple ".g" format ------------------------
    if (!config_file.empty() && config_file.back() == 'g') {
        std::ifstream in(config_file);
        if (!in.is_open()) {
            std::cerr << "Error: Unable to open configure file: " << config_file << "\n";
            std::exit(1);
        }
        for (int i = 1; i <= agent_num; ++i) {
            long long x1, y1, x2, y2;
            in >> x1 >> y1 >> x2 >> y2;   // file gives x y x y (per your branch)
            if (!in) { std::cerr << "[read_scenario] EOF while reading .g line " << i << "\n"; break; }

            // Preferred interpretation (x,y)
            int sx = (int)x1, sy = (int)y1;
            int gx = (int)x2, gy = (int)y2;

            bool okS = in_bounds(sx, sy);
            bool okG = in_bounds(gx, gy);

            // Try a single swap if needed
            if (!okS || !okG) {
                if (!okS) print_oob(i, "start", sx, sy);
                if (!okG) print_oob(i, "goal",  gx, gy);
                std::swap(sx, sy);
                std::swap(gx, gy);
                bool okS2 = in_bounds(sx, sy);
                bool okG2 = in_bounds(gx, gy);
                if (!(okS2 && okG2)) {
                    if (!okS2) print_oob(i, "start(swapped)", sx, sy);
                    if (!okG2) print_oob(i, "goal(swapped)",  gx, gy);
                    throw std::out_of_range("[read_scenario] irrecoverable OOB in .g file");
                } else {
                    std::cerr << "[read_scenario] NOTE: used swapped (x,y) on .g line " << i << "\n";
                }
            }

            auto [rx1, ry1] = to_refined(sx, sy, refine_R);
            auto [rx2, ry2] = to_refined(gx, gy, refine_R);
            start_end.emplace_back(map.getID(rx1, ry1), map.getID(rx2, ry2));
        }
        return;
    }

    // ------------------------ Case 2: MovingAI-like ".scen" ---------------------
    std::ifstream in(config_file);
    if (!in.is_open()) {
        std::cerr << "Error: Unable to open configure file: " << config_file << "\n";
        std::exit(1);
    }

    // Header: "version <int>"
    std::string version_tag; int version_num = 0;
    in >> version_tag >> version_num; // e.g., "version 1"
    if (!in) { std::cerr << "[read_scenario] malformed header in " << config_file << "\n"; std::exit(1); }

    // Each subsequent line (MovingAI): idx map W H sy sx gy gx cost
    // We will *parse* W/H and compare to map.width/map.height, then interpret (x,y) carefully.
    for (int i = 1; i <= agent_num; ++i) {
        int idx = -1;
        std::string map_name;
        int Wf = -1, Hf = -1;
        long long sy, sx, gy, gx;
        double cost = 0.0;

        in >> idx >> map_name >> Wf >> Hf >> sy >> sx >> gy >> gx >> cost;
        if (!in) { std::cerr << "[read_scenario] EOF while reading line " << i << "\n"; break; }

        if (Wf != map.width || Hf != map.height) {
            std::cerr << "[read_scenario] dims mismatch on line " << i
                      << " : scen " << Wf << "x" << Hf
                      << " vs map " << map.width << "x" << map.height << "\n";
            // We continue, but this is a strong hint the (x,y) interpretation may need a swap.
        }

        // MovingAI lines provide (sy, sx, gy, gx) i.e., (y,x) pairs.
        // First try canonical (x = sx, y = sy):
        int sx1 = (int)sx, sy1 = (int)sy;
        int gx1 = (int)gx, gy1 = (int)gy;
        bool okS = in_bounds(sx1, sy1);
        bool okG = in_bounds(gx1, gy1);

        if (!okS || !okG) {
            if (!okS) print_oob(i, "start", sx1, sy1);
            if (!okG) print_oob(i, "goal",  gx1, gy1);

            // Try swap once (x=sy, y=sx)
            int sx2 = (int)sy, sy2 = (int)sx;
            int gx2 = (int)gy, gy2 = (int)gx;
            bool okS2 = in_bounds(sx2, sy2);
            bool okG2 = in_bounds(gx2, gy2);
            if (!(okS2 && okG2)) {
                if (!okS2) print_oob(i, "start(swapped)", sx2, sy2);
                if (!okG2) print_oob(i, "goal(swapped)",  gx2, gy2);
                throw std::out_of_range("[read_scenario] irrecoverable OOB in .scen line");
            } else {
                std::cerr << "[read_scenario] WARNING: needed (x,y) swap on line " << i
                          << " (treated tokens as sx,sy,gx,gy)\n";
                sx1 = sx2; sy1 = sy2; gx1 = gx2; gy1 = gy2;
            }
        }

        auto [rx1, ry1] = to_refined(sx1, sy1, refine_R);
        auto [rx2, ry2] = to_refined(gx1, gy1, refine_R);
        start_end.emplace_back(map.getID(rx1, ry1), map.getID(rx2, ry2));
    }
}



void PreProcessor::build_start_goal_from_coords(Map& map,
                                                const std::vector<std::pair<int,int>>& start_xy,
                                                const std::vector<std::pair<int,int>>& goal_xy,
                                                std::vector<std::pair<size_t, size_t>>& start_end,
                                                int refine_R)
{
    refine_R = 1;
    auto to_refined = [&](int x, int y, int R){
        if (R <= 1) return std::pair<int,int>(x,y);
        return std::pair<int,int>(x*R + R/2, y*R + R/2);
    };

    auto in_bounds = [&](int x, int y){
        return (0 <= x && x < map.width) && (0 <= y && y < map.height);
    };

    start_end.clear();
    const int n = static_cast<int>(std::min(start_xy.size(), goal_xy.size()));
    for (int i = 0; i < n; ++i) {
        int sx1 = start_xy[static_cast<size_t>(i)].first;
        int sy1 = start_xy[static_cast<size_t>(i)].second;
        int gx1 = goal_xy[static_cast<size_t>(i)].first;
        int gy1 = goal_xy[static_cast<size_t>(i)].second;

        if (!in_bounds(sx1, sy1) || !in_bounds(gx1, gy1)) {
            std::cerr << "[build_start_goal_from_coords] OOB agent " << i << "\n";
            throw std::out_of_range("build_start_goal_from_coords OOB");
        }

        auto [rx1, ry1] = to_refined(sx1, sy1, refine_R);
        auto [rx2, ry2] = to_refined(gx1, gy1, refine_R);
        start_end.emplace_back(map.getID(rx1, ry1), map.getID(rx2, ry2));
    }
}

void PreProcessor::cost_init(Map& map, std::vector<Edge>& edges, int dim)
{
    for(int x = 0; x < map.height; x ++) {
        for(int y = 0; y < map.width; y ++){
            if(map.getVal(x, y) == -1){
                continue;
            }
            std::vector<size_t> cost(dim, 0);
            edges.push_back(Edge(map.getID(x, y), map.getID(x, y), std::vector<size_t>(cost)));        //  add remain place action

            if(y < map.width-1 && map.getVal(x, y+1) == 0){
                edges.push_back(Edge(map.getID(x, y+1), map.getID(x, y), std::vector<size_t>(cost)));
            }
            if(y > 0 && map.getVal(x, y-1) == 0){
                edges.push_back(Edge(map.getID(x, y-1), map.getID(x, y), std::vector<size_t>(cost)));
            }
            if(x < map.height-1 && map.getVal(x+1, y) == 0){
                edges.push_back(Edge(map.getID(x+1, y), map.getID(x, y), std::vector<size_t>(cost)));
            }
            if(x > 0 && map.getVal(x-1, y) == 0){
                edges.push_back(Edge(map.getID(x-1, y), map.getID(x, y), std::vector<size_t>(cost)));
            }
            // also add the diagonals
            if(x < map.height-1 && y < map.width-1 && map.getVal(x+1, y+1) == 0){
                edges.push_back(Edge(map.getID(x+1, y+1), map.getID(x, y), std::vector<size_t>(cost)));
            }
            if(x < map.height-1 && y > 0 && map.getVal(x+1, y-1) == 0){
                edges.push_back(Edge(map.getID(x+1, y-1), map.getID(x, y), std::vector<size_t>(cost)));
            }
            if(x > 0 && y < map.width-1 && map.getVal(x-1, y+1) == 0){
                edges.push_back(Edge(map.getID(x-1, y+1), map.getID(x, y), std::vector<size_t>(cost)));
            }
            if(x > 0 && y > 0 && map.getVal(x-1, y-1) == 0){
                edges.push_back(Edge(map.getID(x-1, y-1), map.getID(x, y), std::vector<size_t>(cost)));
            }
        }
    }
}

void PreProcessor::read_cost(std::string cost_file, Map& map, std::vector<Edge>& edges, int dim_i)
{   
    std::ifstream Input(cost_file);
    if(!Input.is_open()){
        std::cout << "Error: Unable to open cost file." << std::endl;
        exit(1);
    }
    Map cost_map(map.width, map.height);

    // Read the map data from the file and store it in the Map object
    for(int x = 0; x < map.height; x ++) {
        for(int y = 0; y < map.width; y ++){
            size_t cost;
            Input >> cost;
            // std::cout << cost1 << std::endl;
            cost_map.setVal(x, y, cost);
        }
    }

    for(int x = 0; x < map.height; x ++) {
        for(int y = 0; y < map.width; y ++){
            if(map.getVal(x, y) == -1){
                continue;
            }
            for(Edge& edge: edges){
                if(edge.target == map.getID(x, y)){
                    edge.cost.at(dim_i) = cost_map.getVal(x, y);
                }
            }
        }
    }
    Input.close();
}
