// For simplier access
::NUM_TRAVERSE_TYPES <- Constants.ENavTraverseType.NUM_TRAVERSE_TYPES
::FLT_MAX <- 3.402823466e+38

// Constrains an angle into [-180, 180] range
::NormalizeAngle <- function(target)
{
	target %= 360.0
	if (target > 180.0)
		target -= 360.0
	else if (target < -180.0)
		target += 360.0
	return target
}

// Approaches an angle at a given speed
::ApproachAngle <- function(target, value, speed)
{
	target = NormalizeAngle(target)
	value = NormalizeAngle(value)
	local delta = NormalizeAngle(target - value)
	if (delta > speed)
		return value + speed
	else if (delta < -speed)
		return value - speed
	return value
}

// Converts a vector direction into angles
::VectorAngles <- function(forward)
{
	local yaw, pitch
	if (forward.y == 0.0 && forward.x == 0.0)
	{
		yaw = 0.0
		if (forward.z > 0.0)
			pitch = 270.0
		else
			pitch = 90.0
	}
	else
	{
		yaw = (atan2(forward.y, forward.x) * 180.0 / Constants.Math.Pi)
		if (yaw < 0.0)
			yaw += 360.0
		pitch = (atan2(-forward.z, forward.Length2D()) * 180.0 / Constants.Math.Pi)
		if (pitch < 0.0)
			pitch += 360.0
	}

	return QAngle(pitch, yaw, 0.0)
}

// Coordinate which is part of a path
class ::PathPoint
{
	constructor(_area, _pos, _how)
	{
		area = _area
		pos = _pos
		how = _how
	}

	area = null		// Which area does this point belong to?
	pos = null		// Coordinates of the point
	how = null		// Type of traversal. See Constants.ENavTraverseType
}

// The big boy that handles all our behavior
class ::Bot
{
	constructor(spawn_origin, target = null)
	{
		bot = SpawnEntityFromTable("base_boss",
		{
			targetname = "bot",
			origin = spawn_origin,
			model = "models/bots/heavy/bot_heavy.mdl",
			playbackrate = 1.0, // Required for animations to be simulated
			// The following is done to prevent default base_boss death behavior
			// Set the health to something really big
			health = FLT_MAX
		})
		// Track the health manually by using npc_hurt event and fire our custom death
		health = 300

		// Fix the default step height which is too high
		bot.AcceptInput("SetStepHeight", "18", null, null)

		// Add scope to the entity
		bot.ValidateScriptScope()
		local scope = bot.GetScriptScope()
		// Append custom bot functionality
		scope.bot_brain <- this

		// Add behavior that will run every tick
		scope.Think <- function() {
			// Let this class handle all the work
			return bot_brain.Update()
		}
		AddThinkToEnt(bot, "Think")


		bot_pos = spawn_origin

		move_speed = 230.0
		turn_rate = 5.0
		search_dist_z = 128.0
		search_dist_nearest = 128.0

		path = []
		path_index = 0
		path_count = 0
		path_reach_dist = 16.0

		path_update_time_next = Time()
		path_update_time_delay = 0.2
		path_closest_distance = 100.0
		path_follow_ent_dist = 50.0

		area_list = {}

		// If the destination hasn't been set yet make it be at the bot's spawn location
		if (target == null)
			target = spawn_origin

		SetDestination(target)


		seq_idle = bot.LookupSequence("Stand_MELEE")
		seq_run = bot.LookupSequence("Run_MELEE")
		pose_move_x = bot.LookupPoseParameter("move_x")


		debug = true
	}

	function SetDestination(target)
	{
		// If our destination is an entity
		if (typeof target == "instance")
		{
			path_follow_ent = target
			// path_target_pos will be calculated in UpdatePath()
		}
		// If it's a vector
		else
		{
			path_follow_ent = null
			path_target_pos = target
		}

		// Force a path update on this frame
		if (UpdatePath())
			SetDirection(path[0].pos)
	}

	function SetDirection(pos)
	{
		move_pos = pos
		// Direction towards path point
		move_dir = move_pos - bot_pos
		move_dir.Norm()

		// Conversion from direction into QAngle form to calculate the bot rotation angles
		move_ang = VectorAngles(move_dir)
	}

	function UpdatePath()
	{
		// Clear out the path first
		ResetPath()

		// If there is a target entity specified, then the bot will pathfind to the entity
		if (path_follow_ent && path_follow_ent.IsValid())
			path_target_pos = path_follow_ent.GetOrigin()

		// Pathfind from the bot's position to the target position
		local pos_start = bot_pos
		local pos_end = path_target_pos

		local area_start = NavMesh.GetNavArea(pos_start, search_dist_z)
		local area_end = NavMesh.GetNavArea(pos_end, search_dist_z)

		// If either area was not found, try use the closest one
		if (area_start == null)
		{
			area_start = NavMesh.GetNearestNavArea(pos_start, search_dist_nearest, false, true)
			// If either area is still missing, then bot can't progress
			if (area_start == null)
				return false
		}
		if (area_end == null)
		{
			area_end = NavMesh.GetNearestNavArea(pos_end, search_dist_nearest, false, true)
			if (area_end == null)
				return false
		}

		// If the start and end area is the same, one path point is enough and all the expensive path building can be skipped
		if (area_start == area_end)
		{
			// From bot's origin
			path.append(PathPoint(area_start, pos_start, NUM_TRAVERSE_TYPES))
			// To target's position
			path.append(PathPoint(area_end, pos_end, NUM_TRAVERSE_TYPES))
			path_count = 2

			// For the debug mode
			area_list["area0"] <- area_start

			return true
		}

		// Build list of areas required to get from the start to the end
		if (!NavMesh.GetNavAreasFromBuildPath(area_start, area_end, pos_end, 0.0, Constants.ETFTeam.TEAM_ANY, false, area_list))
			return false

		local area_count = area_list.len()
		// No areas found? Uh oh
		if (area_count == 0)
			return false

		// First point is simply our current position
		path.append(PathPoint(area_start, pos_start, NUM_TRAVERSE_TYPES))

		// Now build points using the list of areas, which the bot will then target
		// The areas are built from the end to the start so we need a reversed iteration to build the path points
		for (local i = area_count - 1; i >= 0; i--)
		{
			local area = area_list["area" + i]
			path.append(PathPoint(area, area.GetCenter(), area.GetParentHow()))
		}

		// Now compute accurate path points, using adjacent points + direction data from nav
		path_count = path.len()
		for (local i = 1; i < path_count; i++)
		{
			local point_from = path[i - 1]
			local point_to = path[i]

			// Computes closest point within the "portal" between adjacent areas
			point_to.pos = point_from.area.ComputeClosestPointInPortal(point_to.area, point_to.how, point_from.pos)
		}

		// Add a final point so the bot can precisely move towards the end point when it reaches the final area
		path.append(PathPoint(area_end, pos_end, NUM_TRAVERSE_TYPES))
		path_count++

		return true
	}

	function AdvancePath()
	{
		// Check for valid path first
		if (path_count == 0)
			return false

		// If we're close enough to the target stop to not push our target entity
		if (path_follow_ent && path_follow_ent.IsValid() &&
			(path_target_pos - bot_pos).Length() < path_closest_distance)
		{
			ResetPath()
			return false
		}

		// Are we close enough to the path point to consider it as 'reached'?
		if ((move_pos - bot_pos).Length2D() < path_reach_dist)
		{
			// Start moving to the next point
			path_index++
			if (path_index >= path_count)
			{
				// End of the line!
				ResetPath()
				return false
			}

			SetDirection(path[path_index].pos)
		}

		return true
	}

	function ResetPath()
	{
		area_list.clear()
		path.clear()
		path_count = 0
		path_index = 0
	}

	function Move()
	{
		// Recompute path to our target if present
		if (path_follow_ent && path_follow_ent.IsValid())
		{
			// Is it time to re-compute the path?
			local time = Time()
			if (path_update_time_next < time)
			{
				// Check if target has moved far away enough
				local follow_ent_pos = path_follow_ent.GetOrigin()
				if ((path_target_pos - follow_ent_pos).Length() > path_follow_ent_dist &&
					(bot_pos - follow_ent_pos).Length() > path_closest_distance)
				{
					if (UpdatePath())
						SetDirection(path[0].pos)
					// Don't recompute again for a moment
					path_update_time_next = time + path_update_time_delay
				}
			}
		}

		// Check and advance up our path
		if (AdvancePath())
		{
			// Set our new position
			// Velocity is calculated from direction times speed, and converted from per-second to per-tick time
			bot.SetAbsOrigin(bot_pos + (move_dir * move_speed * FrameTime()))

			// Visualize current path in debug mode
			if (debug)
			{
				// Stay around for 1 tick
				// Debugoverlays are created on 1st tick but start rendering on 2nd tick, hence this must be doubled
				local frame_time = FrameTime() * 2.0

				// Draw connected path points
				local path_start_index = path_index
				if (path_start_index == 0)
					path_start_index++

				for (local i = path_start_index; i < path_count; i++)
				{
					DebugDrawLine(path[i - 1].pos, path[i].pos, 0, 255, 0, true, frame_time)
				}

				// Draw areas from built path
				foreach (name, area in area_list)
				{
					area.DebugDrawFilled(255, 0, 0, 30, frame_time, true, 0.0)
					DebugDrawText(area.GetCenter(), name, false, frame_time)
				}
			}

			return true
		}

		return false
	}


	function Update()
	{
		bot_pos = bot.GetOrigin()

		// Try moving
		if (Move())
		{
			// Moving, set the run animation
			if (bot.GetSequence() != seq_run)
			{
				bot.SetSequence(seq_run)
				bot.SetPoseParameter(pose_move_x, 1.0) // Set the move_x pose to max weight
			}
		}
		else
		{
			// Not moving, set the idle animation
			if (bot.GetSequence() != seq_idle)
			{
				bot.SetSequence(seq_idle)
				bot.SetPoseParameter(pose_move_x, 0.0) // Clear the move_x pose
			}

			// If the bot is standing still, look at the target instead of the path points
			if (path_follow_ent && path_follow_ent.IsValid())
				SetDirection(path_follow_ent.GetOrigin())
		}

		// Rotating the bot
		// Approach new desired angle but only on the Y axis
		local bot_ang = bot.GetAbsAngles()
		bot_ang.y = ApproachAngle(move_ang.y, bot_ang.y, turn_rate)
		// Set our new angles
		bot.SetAbsAngles(bot_ang)


		// Replay animation if it has finished
		if (bot.GetCycle() > 0.99)
			bot.SetCycle(0.0)

		// Run animations
		bot.StudioFrameAdvance()
		bot.DispatchAnimEvents(bot)


		return -1.0 // Think again next frame
	}

	function OnKilled()
	{
		// Change life state to "dying"
		// The bot won't take any more damage, and sentries will stop targeting it
		NetProps.SetPropInt(bot, "m_lifeState", 1)
		// For this example, turn into a ragdoll with the saved damage force
		bot.BecomeRagdollOnClient(damage_force)
		// Stop pathfinding
		AddThinkToEnt(bot, null)

		// Custom death behavior can be added here
	}

	bot = null						// The bot entity we belong to
	health = null					// Manual track of health to prevent default base_boss death behavior


	move_speed = null				// How fast to move
	turn_rate = null				// How fast to turn
	search_dist_z = null			// Maximum distance to look for a nav area downwards
	search_dist_nearest = null 		// Maximum distance to look for any nearby nav area

	bot_pos = null					// Origin of the bot
	move_pos = null					// Current target destination (path point)
	move_dir = null					// Current move direction
	move_ang = null					// Current move direction in angle form

	path = null						// List of BotPathPoints
	path_index = null				// Current path point bot is at, -1 if none
	path_count = null				// Number of path points

	path_follow_ent = null			// What entity to move towards
	path_follow_ent_dist = null		// Maximum distance after which the path is recomputed
									// if target entity's current position is too far from our target position
	path_closest_distance = null	// The closest the bot can get to an entity before stopping
									// required to not push the entity when we get too close

	path_reach_dist = null			// Distance to a path point to be considered as 'reached'
	path_target_pos = null			// Position where bot wants to navigate to
	path_update_time_next = null	// Timer for when to update path again
	path_update_time_delay = null   // Seconds to wait before trying to attempt to update path again
	area_list = null				// List of areas built in path

	seq_idle = null					// Animation to use when idle
	seq_run = null					// Animation to use when running
	pose_move_x = null				// Pose parameter to set for running animation

	damage_force = null				// Damage force from the bot's last OnTakeDamage event

	debug = null					// When true, debug visualization is enabled
}

::BotCreate <- function()
{
	// Find point where player is looking
	local player = GetListenServerHost()
	local eye_pos = player.EyePosition()
	local trace =
	{
		start = eye_pos,
		end = eye_pos + (player.EyeAngles().Forward() * 32768.0),
		ignore = player
	}

	if (!TraceLineEx(trace))
	{
		printl("Invalid bot spawn location")
		return null
	}

	// Spawn bot at the end point and start following the player
	return Bot(trace.pos, player)
}

 // event listener (see Listening for Events wexample)
// CollectEventsInScope
// ({
// 	function OnScriptHook_OnTakeDamage(params)
// 	{
// 		local victim = params.const_entity
// 		local scope = victim.GetScriptScope()

// 		if (victim.IsPlayer() && "bot_brain" in params.inflictor.GetScriptScope()
// 			&& params.damage_type == Constants.FDmgType.DMG_CRUSH)
// 		{
// 			// Don't crush the player if a bot pushes them into a wall
// 			params.damage = 0.0
// 		}

// 		if ("bot_brain" in scope)
// 		{
// 			// Save the damage force into the bot's data
// 			scope.bot_brain.damage_force = params.damage_force
// 		}
// 	}

// 	function OnGameEvent_npc_hurt(params)
// 	{
// 		local victim = EntIndexToHScript(params.entindex)
// 		local scope = victim.GetScriptScope()

// 		if ("bot_brain" in scope)
// 		{
// 			// Substract the damage dealt from our manual health track
// 			scope.bot_brain.health -= params.damageamount
// 			// Check if a bot is about to die
// 			if (scope.bot_brain.health <= 0.0)
// 			{
// 				// Run the bot's OnKilled function
// 				scope.bot_brain.OnKilled()
// 			}
// 		}
// 	}
// })
::LastBot <- BotCreate()

// to dynamically change the bot's destination you can write the following in the console
// script LastBot.SetDestination(Vector(100.0, 100.0, 0.0))
// script LastBot.SetDestination(GetListenServerHost())
// script LastBot.SetDestination(GetListenServerHost().GetOrigin())
// etc
