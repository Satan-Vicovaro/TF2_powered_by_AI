// Globals
::tracker_logic <- null;
::FLT_MAX <- 3.402823466e+38;
::distances <- {};
::min_positions <- {};
::min_positions_diffs <- {};
::track_iterations <- 0;
::TRACKING_RATE <- 0.01; // seconds between updates
::TF_TEAM_BLUE <- 3
::projectile_destroyed <- {};  // ownerIndex → bool
::projectile_classes <- ["tf_projectile_rocket", "tf_projectile_pipe"];

const stress_testing = true;
const debug = true;

IncludeScript("file_scripts");

function closest_point_on_bbox(pos, ent)
{
    local origin = ent.GetLocalOrigin();
    local mins = ent.GetBoundingMins();
    local maxs = ent.GetBoundingMaxs();

    local clamped = Vector(
        clamp(pos.x, origin.x + mins.x, origin.x + maxs.x),
        clamp(pos.y, origin.y + mins.y, origin.y + maxs.y),
        clamp(pos.z, origin.z + mins.z, origin.z + maxs.z)
    );

    return clamped;
}

function clamp(val, min, max)
{
    return (val < min) ? min : (val > max) ? max : val;
}

function calculate_distance(pos_A, pos_B)
{
    local dx = pos_A.x - pos_B.x;
    local dy = pos_A.y - pos_B.y;
    local dz = pos_A.z - pos_B.z;

    return sqrt(dx*dx + dy*dy + dz*dz);
}

function get_target_bot_entity()
{
    local ent = null;
    while (ent = Entities.FindByClassname(ent, "player"))
    {
        if (ent == GetListenServerHost()) continue;
        if (ent.GetTeam() == ::TF_TEAM_BLUE) {
            return ent;
        }
    }
    return null;
}

function vector_to_string(vec)
{
    return format("%.3f %.3f %.3f", vec.x, vec.y, vec.z);
}

function send_projectile_info()
{
    if(debug) printl("sending info");

    local projectile_info = ""
    local source_map = ::min_positions_diffs

    if(source_map.len() == 0)
    {
        projectile_info = "b none\n";
    }
    else
    {
        foreach(ownerIndex, data in source_map)
        {
            projectile_info += format("b %d %s\n", ownerIndex, vector_to_string(data));
        }
    }

    append_to_file("squirrel_out", projectile_info);

    // Cleanup after saving
    ::distances.clear();
    ::min_positions.clear();
    ::min_positions_diffs.clear();
    ::projectile_destroyed.clear();
    if (debug) printl("Tracking data cleared after write");
}

// === TRACKING LOGIC ===

function start_tracking()
{
    ::distances.clear();
    ::min_positions.clear();
    ::min_positions_diffs.clear();

    if ("tracker_logic" in getroottable() && ::tracker_logic != null && ::tracker_logic.IsValid())
    {
        NetProps.SetPropString(::tracker_logic, "m_iszScriptThinkFunction", "");
        ::tracker_logic.Destroy();
        ::tracker_logic <- null;
    }

    ::tracker_logic = Entities.CreateByClassname("logic_script");
    ::tracker_logic.ValidateScriptScope();
    ::tracker_logic.KeyValueFromString("targetname", "projectile_tracker");

    local scope = ::tracker_logic.GetScriptScope();
    scope["TrackThink"] <- TrackThink;

    AddThinkToEnt(::tracker_logic, "TrackThink");
}

function mark_projectile_destroyed(ownerIndex)
{
    ::projectile_destroyed[ownerIndex] <- true;
    if (debug) printl("Marked projectile destroyed for owner " + ownerIndex);
}

function stop_tracking()
{
    if ("tracker_logic" in getroottable() && ::tracker_logic != null && ::tracker_logic.IsValid())
    {
        NetProps.SetPropString(::tracker_logic, "m_iszScriptThinkFunction", "");
        ::tracker_logic.Destroy();
        ::tracker_logic <- null;
        printl("Tracking stopped.");
        ::track_iterations <- 0;
    }
    else
    {
        printl("No active tracker to stop.");
    }
}


// The thinker function, runs every TRACKING_RATE seconds
function TrackThink()
{
    ::track_iterations++;
    local target_ent = get_target_bot_entity();
    if (!target_ent) return ::TRACKING_RATE;
    foreach (classname in ::projectile_classes)
    {
        local ent = null;
        
        while (ent = Entities.FindByClassname(ent, classname))
        {
            if (!ent || !ent.IsValid()) continue;

            local owner = null;
            try {owner = ent.GetOwner(); } catch(e) {printl("owner error" + e)}

            if (owner == null || !owner.IsValid())
            {
                if (debug) printl("Projectile has no valid owner yet, skipping.");
                continue;
            }

            local ownerIndex = owner.entindex();
            local pos = ent.GetOrigin();
            local target_pos = closest_point_on_bbox(pos, target_ent);
            local currDistance = calculate_distance(pos, target_pos);
            local pos_diff = target_pos - pos;
            
            // Reset if projectile was destroyed and new projectile detected
            if ((ownerIndex in ::projectile_destroyed) && ::projectile_destroyed[ownerIndex] == true)
            {
                ::distances[ownerIndex] <- currDistance;
                ::min_positions[ownerIndex] <- pos;
                ::min_positions_diffs[ownerIndex] <- pos_diff;
                ::projectile_destroyed[ownerIndex] <- false;

                if (debug) printl("Resetting data for owner " + ownerIndex + " due to destroyed projectile.");
            }
            // Otherwise update if smaller distance or untracked yet
            else if (!(ownerIndex in ::distances) || currDistance <= ::distances[ownerIndex])
            {
                ::distances[ownerIndex] <- currDistance;
                ::min_positions[ownerIndex] <- pos;
                ::min_positions_diffs[ownerIndex] <- pos_diff;
            }

            if(debug) {
                printl("[Tracking] Owner " + ownerIndex + " | Target: " + target_pos + " | Projectile: " + pos + " | Dist: " + currDistance + " | Min dist: " + ::distances[ownerIndex]);
            }
            
        }
    }

    if(stress_testing) {
            if (::track_iterations % 1000 == 0) {
                log_tracking_table_sizes();             
            }
        }

    return ::TRACKING_RATE;
}

function log_tracking_table_sizes()
{
    local sizes = "";
    sizes += format("distances: %d\n", ::distances.len());
    sizes += format("min_positions: %d\n", ::min_positions.len());
    sizes += format("min_positions_diffs: %d\n", ::min_positions_diffs.len());
    sizes += format("projectile_destroyed: %d\n", ::projectile_destroyed.len());

    append_to_file("sizes.txt", "[Table Sizes @ " + ::track_iterations + "]\n" + sizes);
}

local EventsID = UniqueString()
getroottable()[EventsID] <-
{
	OnScriptHook_KillProjectileHooks = function(_) {
        printl("Deleting hooks for projectile events")
        delete getroottable()[EventsID]
    }

	OnGameEvent_projectile_direct_hit = function(params) {
        local ownerIndex = params.attacker;
        if (debug) printl("Projectile direct hit: attacker = " + ownerIndex);

        mark_projectile_destroyed(ownerIndex);
    }

	OnGameEvent_projectile_removed = function(params) {
        local ownerIndex = params.attacker;
        if (debug) printl("Projectile removed: attacker = " + ownerIndex);

        mark_projectile_destroyed(ownerIndex);
    }
}

local EventsTable = getroottable()[EventsID]
foreach (name, callback in EventsTable) EventsTable[name] = callback.bindenv(this)
__CollectGameEventCallbacks(EventsTable)
