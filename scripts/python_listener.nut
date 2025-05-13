class :: python_listener {
    out_file_path = null;
    in_file_path = null;

    listener_bot = null;
    constructor(){
        out_file_path = "squirrel_out";
        in_file_path = "squirrel_in";

		bot = SpawnEntityFromTable("base_boss",
		{
			targetname = "bot",
			origin = Vector(1700,1700,0),
			model = "models/bots/skeleton_sniper/skeleton_sniper.mdl",
			playbackrate = 1.0, // Required for animations to be simulated
			// The following is done to prevent default base_boss death behavior
			// Set the health to something really big
			health = FLT_MAX
		})
		// Track the health manually by using npc_hurt event and fire our custom death
		health = 300
    }
}

p_listener <- python_listener()
