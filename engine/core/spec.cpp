#include "core/spec.hpp"

#include <fstream>
#include <stdexcept>

static void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

PipelineSpec load_spec(const std::string& json_path) {
    std::ifstream f(json_path.c_str());
    require(f.good(), "cannot open pipeline.json: " + json_path);

    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("pipeline.json parse error: ") + e.what());
    }

    require(j.contains("version"), "pipeline.json missing required field: version");
    require(j["version"].is_number_integer(), "pipeline.json field 'version' must be an integer");

    PipelineSpec spec;
    spec.version = j["version"].get<int>();
    require(spec.version == 1,
            "pipeline.json unsupported version: " + std::to_string(spec.version) +
                " (this engine speaks version 1; field: version)");
    if (j.contains("name")) spec.name = j["name"].get<std::string>();
    if (j.contains("target") && j["target"].contains("core_mask")) {
        spec.core_mask = j["target"]["core_mask"].get<std::string>();
    }

    if (j.contains("models")) {
        require(j["models"].is_object(), "pipeline.json field 'models' must be an object");
        for (auto it = j["models"].begin(); it != j["models"].end(); ++it) {
            const nlohmann::json& m = it.value();
            require(m.contains("family"), "models." + it.key() + " missing field: family");
            require(m.contains("file"), "models." + it.key() + " missing field: file");
            ModelSpec ms;
            ms.family = m["family"].get<std::string>();
            ms.file = m["file"].get<std::string>();
            if (m.contains("input")) ms.input = m["input"];
            spec.models[it.key()] = ms;
        }
    }

    require(j.contains("pipeline"), "pipeline.json missing required field: pipeline");
    require(j["pipeline"].is_array(), "pipeline.json field 'pipeline' must be an array");
    std::map<std::string, bool> seen_ids;
    for (const auto& s : j["pipeline"]) {
        require(s.contains("id"), "pipeline stage missing field: id");
        require(s.contains("plugin"), "pipeline stage '" + s.value("id", "?") + "' missing field: plugin");
        StageSpec ss;
        ss.id = s["id"].get<std::string>();
        require(!seen_ids.count(ss.id), "pipeline.json duplicate stage id: " + ss.id);
        seen_ids[ss.id] = true;
        ss.plugin = s["plugin"].get<std::string>();
        if (s.contains("model")) {
            ss.model = s["model"].get<std::string>();
            require(spec.models.count(ss.model) == 1,
                    "stage '" + ss.id + "' references unknown model key: " + ss.model);
        }
        if (s.contains("params")) ss.params = s["params"];
        spec.pipeline.push_back(ss);
    }
    return spec;
}
